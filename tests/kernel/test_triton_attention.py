"""Numerical correctness for the Triton paged attention kernel.

Compares `_launch_paged_attn` against an eager reference for the four shapes
we ship: prefill no-cache, prefill with cached prefix, decode, page_size>1.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import pytest
import torch
import torch.nn.functional as F

from minisgl.attention.triton import _launch_paged_attn


def _require_gpu():
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")


def _build_paged_cache(
    k_per: torch.Tensor,  # [bs, k_len, num_kv, hd]
    v_per: torch.Tensor,
    page_size: int,
    extra_pages: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack per-batch K/V into a paged cache and emit a [bs, max_pages] page table.

    Page IDs are scrambled (not in batch order) so the kernel really has to follow
    the table — a contiguous identity layout would mask indexing bugs.
    """
    bs, k_len, num_kv, hd = k_per.shape
    pages_per_seq = (k_len + page_size - 1) // page_size
    total_pages = bs * pages_per_seq + extra_pages
    device, dtype = k_per.device, k_per.dtype

    k_cache = torch.zeros(total_pages, page_size, num_kv, hd, dtype=dtype, device=device)
    v_cache = torch.zeros_like(k_cache)
    page_table = torch.zeros(bs, pages_per_seq, dtype=torch.int32, device=device)

    perm = torch.randperm(total_pages, device=device)
    for b in range(bs):
        for p in range(pages_per_seq):
            phys = int(perm[b * pages_per_seq + p].item())
            page_table[b, p] = phys
            for s in range(page_size):
                t = p * page_size + s
                if t < k_len:
                    k_cache[phys, s] = k_per[b, t]
                    v_cache[phys, s] = v_per[b, t]
    return k_cache, v_cache, page_table


def _eager_reference(
    q_per: torch.Tensor,  # [bs, q_len, num_qo, hd]
    k_per: torch.Tensor,  # [bs, k_len, num_kv, hd]
    v_per: torch.Tensor,
    cached_len: int,
) -> torch.Tensor:
    """Manual MHA/GQA with causal mask offset by `cached_len`. Returns [bs, q_len, num_qo, hd]."""
    bs, q_len, num_qo, hd = q_per.shape
    _, k_len, num_kv, _ = k_per.shape
    assert k_len == cached_len + q_len
    group = num_qo // num_kv

    # broadcast K/V across GQA groups
    k_b = k_per.repeat_interleave(group, dim=2)  # [bs, k_len, num_qo, hd]
    v_b = v_per.repeat_interleave(group, dim=2)

    qf = q_per.float().permute(0, 2, 1, 3)  # [bs, h, q_len, hd]
    kf = k_b.float().permute(0, 2, 1, 3)
    vf = v_b.float().permute(0, 2, 1, 3)
    scale = hd ** -0.5
    qk = (qf @ kf.transpose(-1, -2)) * scale  # [bs, h, q_len, k_len]

    q_pos = cached_len + torch.arange(q_len, device=q_per.device)
    k_pos = torch.arange(k_len, device=q_per.device)
    causal = q_pos[:, None] >= k_pos[None, :]  # [q_len, k_len]
    qk = qk.masked_fill(~causal, float("-inf"))
    p = qk.softmax(dim=-1)
    out = p @ vf  # [bs, h, q_len, hd]
    return out.permute(0, 2, 1, 3).to(q_per.dtype)


def _flatten_per_batch(t: torch.Tensor) -> torch.Tensor:
    """[bs, len, h, d] -> [bs*len, h, d]."""
    bs, ln, h, d = t.shape
    return t.reshape(bs * ln, h, d)


def _run_case(
    bs: int,
    q_len: int,
    cached_len: int,
    num_qo: int,
    num_kv: int,
    hd: int,
    page_size: int,
    dtype: torch.dtype = torch.float16,
):
    _require_gpu()
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    k_len = q_len + cached_len

    q_per = torch.randn(bs, q_len, num_qo, hd, dtype=dtype, device=device)
    k_per = torch.randn(bs, k_len, num_kv, hd, dtype=dtype, device=device)
    v_per = torch.randn(bs, k_len, num_kv, hd, dtype=dtype, device=device)

    k_cache, v_cache, page_table = _build_paged_cache(k_per, v_per, page_size)
    k_flat = k_cache.flatten(0, 1)
    v_flat = v_cache.flatten(0, 1)

    cu_seqlens_q = torch.tensor(
        [i * q_len for i in range(bs + 1)], dtype=torch.int32, device=device
    )
    seq_lens = torch.full((bs,), k_len, dtype=torch.int32, device=device)
    cached_lens = torch.full((bs,), cached_len, dtype=torch.int32, device=device)

    q_flat = _flatten_per_batch(q_per)

    out = _launch_paged_attn(
        q=q_flat,
        k_cache_flat=k_flat,
        v_cache_flat=v_flat,
        cu_seqlens_q=cu_seqlens_q,
        seq_lens_k=seq_lens,
        cached_lens=cached_lens,
        page_table=page_table,
        max_seqlen_q=q_len,
        page_size=page_size,
        sm_scale=hd ** -0.5,
    )

    ref = _eager_reference(q_per, k_per, v_per, cached_len)
    ref_flat = _flatten_per_batch(ref)

    torch.testing.assert_close(out, ref_flat, atol=2e-2, rtol=2e-2)


def test_prefill_no_cache_mha():
    _run_case(bs=4, q_len=128, cached_len=0, num_qo=8, num_kv=8, hd=64, page_size=1)


def test_prefill_no_cache_gqa():
    _run_case(bs=2, q_len=256, cached_len=0, num_qo=8, num_kv=1, hd=128, page_size=1)


def test_prefill_with_cache():
    _run_case(bs=2, q_len=64, cached_len=128, num_qo=8, num_kv=2, hd=128, page_size=1)


def test_decode_gqa():
    _run_case(bs=8, q_len=1, cached_len=255, num_qo=8, num_kv=1, hd=128, page_size=1)


def test_page_size_16_prefill():
    _run_case(bs=2, q_len=96, cached_len=64, num_qo=8, num_kv=2, hd=128, page_size=16)


def test_page_size_16_decode():
    _run_case(bs=4, q_len=1, cached_len=127, num_qo=8, num_kv=1, hd=128, page_size=16)
