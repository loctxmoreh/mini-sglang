from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
import triton
import triton.language as tl
from minisgl.core import Batch, get_global_ctx

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@triton.jit
def _paged_attn_fwd_kernel(
    Q,
    K,
    V,
    O,
    cu_seqlens_q,
    seq_lens_k,
    cached_lens,
    page_table,
    sm_scale,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_obs,
    stride_oh,
    stride_od,
    stride_pt_b,
    stride_pt_p,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    KV_GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    q_start = tl.load(cu_seqlens_q + pid_b)
    q_end = tl.load(cu_seqlens_q + pid_b + 1)
    q_len = q_end - q_start
    if pid_m * BLOCK_M >= q_len:
        return

    k_len = tl.load(seq_lens_k + pid_b)
    cache_off = tl.load(cached_lens + pid_b)
    kv_head = pid_h // KV_GROUP

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    q_row = (q_start + offs_m).to(tl.int64)
    q_ptrs = (
        Q
        + q_row[:, None] * stride_qbs
        + pid_h * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q_mask = offs_m[:, None] < q_len
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    abs_q_pos = cache_off + offs_m

    for n_start in range(0, k_len, BLOCK_N):
        n_idx = n_start + offs_n
        n_mask = n_idx < k_len

        page_offset = n_idx // PAGE_SIZE
        slot_in_page = n_idx % PAGE_SIZE
        safe_page_offset = tl.where(n_mask, page_offset, 0)
        page_idx = tl.load(
            page_table + pid_b * stride_pt_b + safe_page_offset * stride_pt_p
        )
        token_idx = page_idx.to(tl.int64) * PAGE_SIZE + slot_in_page

        k_ptrs = (
            K
            + token_idx[:, None] * stride_kbs
            + kv_head * stride_kh
            + offs_d[None, :] * stride_kd
        )
        v_ptrs = (
            V
            + token_idx[:, None] * stride_vbs
            + kv_head * stride_vh
            + offs_d[None, :] * stride_vd
        )
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k)) * sm_scale
        causal_mask = abs_q_pos[:, None] >= n_idx[None, :]
        valid = causal_mask & n_mask[None, :]
        qk = tl.where(valid, qk, -float("inf"))

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    safe_l = tl.where(l_i > 0, l_i, 1.0)
    acc = acc / safe_l[:, None]

    o_ptrs = (
        O
        + q_row[:, None] * stride_obs
        + pid_h * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=q_mask)


def _launch_paged_attn(
    q: torch.Tensor,
    k_cache_flat: torch.Tensor,
    v_cache_flat: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens_k: torch.Tensor,
    cached_lens: torch.Tensor,
    page_table: torch.Tensor,
    max_seqlen_q: int,
    page_size: int,
    sm_scale: float,
) -> torch.Tensor:
    total_q, num_qo_heads, head_dim = q.shape
    num_kv_heads = k_cache_flat.shape[1]
    assert num_qo_heads % num_kv_heads == 0
    kv_group = num_qo_heads // num_kv_heads
    bs = page_table.shape[0]

    o = torch.empty_like(q)

    BLOCK_M = 16 if max_seqlen_q == 1 else 64
    BLOCK_N = 32 if head_dim >= 128 else 64

    grid = (triton.cdiv(max_seqlen_q, BLOCK_M), num_qo_heads, bs)

    _paged_attn_fwd_kernel[grid](
        q,
        k_cache_flat,
        v_cache_flat,
        o,
        cu_seqlens_q,
        seq_lens_k,
        cached_lens,
        page_table,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache_flat.stride(0),
        k_cache_flat.stride(1),
        k_cache_flat.stride(2),
        v_cache_flat.stride(0),
        v_cache_flat.stride(1),
        v_cache_flat.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        page_table.stride(0),
        page_table.stride(1),
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        KV_GROUP=kv_group,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )
    return o


@dataclass
class TritonCaptureData(BaseCaptureData):
    cached_lens: torch.Tensor

    @classmethod
    def create(cls, max_bs: int, max_seq_len: int, device: torch.device, **kwargs):
        return cls(
            seq_lens=torch.ones((max_bs,), dtype=torch.int32, device=device),
            positions=torch.zeros((max_bs,), dtype=torch.int32, device=device),
            cu_seqlens_k=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            cu_seqlens_q=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            page_table=torch.zeros((max_bs, max_seq_len), dtype=torch.int32, device=device),
            cached_lens=torch.zeros((max_bs,), dtype=torch.int32, device=device),
            **kwargs,
        )


@dataclass
class TritonMetadata(BaseAttnMetadata):
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    seq_lens: torch.Tensor
    cached_lens: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    page_table: torch.Tensor

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class TritonBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig) -> None:
        ctx = get_global_ctx()
        self.config = config
        self.kvcache = ctx.kv_cache
        self.page_size = ctx.page_size
        self.scale = config.head_dim ** -0.5
        self.capture: TritonCaptureData | None = None
        self.max_graph_bs = 0
        self.capture_bs: List[int] = []

    def _flat_kv(self, layer_id: int):
        k = self.kvcache.k_cache(layer_id)
        v = self.kvcache.v_cache(layer_id)
        return k.flatten(0, 1), v.flatten(0, 1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, TritonMetadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        k_flat, v_flat = self._flat_kv(layer_id)
        return _launch_paged_attn(
            q=q,
            k_cache_flat=k_flat,
            v_cache_flat=v_flat,
            cu_seqlens_q=metadata.cu_seqlens_q,
            seq_lens_k=metadata.seq_lens,
            cached_lens=metadata.cached_lens,
            page_table=metadata.page_table,
            max_seqlen_q=metadata.max_seqlen_q,
            page_size=self.page_size,
            sm_scale=self.scale,
        )

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs
        padded_size = len(reqs)
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_list = [req.cached_len for req in reqs]
        max_seqlen_k = max(seqlens_k)
        max_seqlen_q = max(seqlens_q)

        cpu_kw = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}
        device = self.kvcache.device

        seq_lens = torch.tensor(seqlens_k, **cpu_kw).to(device, non_blocking=True)
        cached_lens = torch.tensor(cached_list, **cpu_kw).to(device, non_blocking=True)
        cu_seqlens_k = (
            torch.tensor([0] + seqlens_k, **cpu_kw)
            .cumsum_(dim=0)
            .to(device, non_blocking=True)
        )

        if max_seqlen_q == 1:
            cu_seqlens_q = torch.arange(
                0, padded_size + 1, device=device, dtype=torch.int32
            )
        elif all(c == 0 for c in cached_list):
            cu_seqlens_q = cu_seqlens_k
        else:
            cu_seqlens_q = (
                torch.tensor([0] + seqlens_q, **cpu_kw)
                .cumsum_(dim=0)
                .to(device, non_blocking=True)
            )

        global_pt = get_global_ctx().page_table
        new_pt = torch.stack(
            [global_pt[req.table_idx, : max_seqlen_k : self.page_size] for req in reqs]
        )
        if self.page_size > 1:
            new_pt.div_(self.page_size, rounding_mode="floor")

        batch.attn_metadata = TritonMetadata(
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seq_lens=seq_lens,
            cached_lens=cached_lens,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            page_table=new_pt,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        self.capture = TritonCaptureData.create(
            max_bs, max_seq_len // self.page_size, self.kvcache.device
        )
        self.max_graph_bs = max_bs
        self.capture_bs = sorted(bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        bs = batch.size
        assert bs in self.capture_bs and self.capture is not None
        c = self.capture
        batch.attn_metadata = TritonMetadata(
            cu_seqlens_q=c.cu_seqlens_q[: bs + 1],
            cu_seqlens_k=c.cu_seqlens_k[: bs + 1],
            seq_lens=c.seq_lens[:bs],
            cached_lens=c.cached_lens[:bs],
            max_seqlen_q=1,
            max_seqlen_k=c.page_table.size(1) * self.page_size,
            page_table=c.page_table[:bs, :],
        )

    def prepare_for_replay(self, batch: Batch) -> None:
        metadata = batch.attn_metadata
        bs = batch.padded_size
        assert isinstance(metadata, TritonMetadata)
        assert self.capture is not None and bs in self.capture_bs
        c = self.capture
        table_len = metadata.page_table.size(1)
        c.cu_seqlens_k[: bs + 1].copy_(metadata.cu_seqlens_k)
        c.seq_lens[:bs].copy_(metadata.seq_lens)
        c.cached_lens[:bs].copy_(metadata.cached_lens)
        c.page_table[:bs, :table_len].copy_(metadata.page_table)
