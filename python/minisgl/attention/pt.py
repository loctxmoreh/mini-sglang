"""Pure PyTorch paged attention backend.

Uses ``torch.nn.functional.scaled_dot_product_attention`` (SDPA) with a loop
over batch items.  Works on both CUDA and ROCm without any compiled custom
kernels.  Slower than FlashInfer / FlashAttention but fully portable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch
import torch.nn.functional as F

from .fa import FACaptureData, FAMetadata, FlashAttentionBackend

if TYPE_CHECKING:
    from minisgl.core import Batch


def _pt_paged_attn(
    q: torch.Tensor,           # [total_tokens, num_qo_heads, head_dim]
    k_cache: torch.Tensor,     # [num_pages, page_size, num_kv_heads, head_dim]
    v_cache: torch.Tensor,     # [num_pages, page_size, num_kv_heads, head_dim]
    page_table: torch.Tensor,  # [batch_size, max_pages_per_seq]  (GPU)
    cache_seqlens: torch.Tensor,  # [batch_size]  (GPU, int32)
    cu_seqlens_q: torch.Tensor,   # [batch_size+1] (GPU, int32)
    page_size: int,
    softmax_scale: float,
) -> torch.Tensor:
    num_qo_heads = q.size(1)
    num_kv_heads = k_cache.size(2)
    head_dim = q.size(2)
    num_groups = num_qo_heads // num_kv_heads
    batch_size = page_table.size(0)

    # Bring index tensors to CPU for the loop.  One sync per forward call.
    cache_seqlens_cpu = cache_seqlens.cpu()
    cu_seqlens_q_cpu = cu_seqlens_q.cpu()
    page_table_cpu = page_table.cpu()

    outputs: List[torch.Tensor] = []

    for i in range(batch_size):
        seqlen_k = int(cache_seqlens_cpu[i])
        q_start = int(cu_seqlens_q_cpu[i])
        q_end = int(cu_seqlens_q_cpu[i + 1])
        seqlen_q = q_end - q_start

        q_i = q[q_start:q_end]  # [seqlen_q, num_qo_heads, head_dim]

        # Gather KV pages for this sequence.
        max_pages = (seqlen_k + page_size - 1) // page_size
        page_ids = page_table_cpu[i, :max_pages].to(k_cache.device)
        # [max_pages * page_size, num_kv_heads, head_dim] → trim to seqlen_k
        k_i = k_cache[page_ids].reshape(-1, num_kv_heads, head_dim)[:seqlen_k]
        v_i = v_cache[page_ids].reshape(-1, num_kv_heads, head_dim)[:seqlen_k]

        # Expand KV heads for GQA.
        if num_groups > 1:
            k_i = k_i.repeat_interleave(num_groups, dim=1)
            v_i = v_i.repeat_interleave(num_groups, dim=1)

        # SDPA expects [batch, heads, seq, dim].
        q_sdpa = q_i.transpose(0, 1).unsqueeze(0)   # [1, H, seqlen_q, D]
        k_sdpa = k_i.transpose(0, 1).unsqueeze(0)   # [1, H, seqlen_k, D]
        v_sdpa = v_i.transpose(0, 1).unsqueeze(0)   # [1, H, seqlen_k, D]

        # ROCm SDPA with is_causal=True produces NaN/large errors when
        # seqlen_q != seqlen_k (non-square attention matrices).  Use explicit
        # dtype-safe bias instead:
        #   - decode (seqlen_q == 1): no mask needed — query attends to all keys
        #   - full prefill (seqlen_q == seqlen_k): is_causal=True is safe
        #   - extend/chunked prefill: explicit lower-triangular bias
        if seqlen_q == 1:
            out_i = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                scale=softmax_scale,
                is_causal=False,
            )
        elif seqlen_q == seqlen_k:
            out_i = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                scale=softmax_scale,
                is_causal=True,
            )
        else:
            # Build causal bias: query[i] attends to key[j] iff j <= offset + i
            offset = seqlen_k - seqlen_q
            attn_bias = torch.zeros(
                1, 1, seqlen_q, seqlen_k, device=q_i.device, dtype=q_i.dtype
            )
            # Mask out future keys
            future_mask = torch.ones(seqlen_q, seqlen_k, device=q_i.device, dtype=torch.bool)
            for qi in range(seqlen_q):
                future_mask[qi, offset + qi + 1:] = False
            attn_bias.masked_fill_(~future_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            out_i = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                attn_mask=attn_bias,
                scale=softmax_scale,
            )

        outputs.append(out_i.squeeze(0).transpose(0, 1))  # [seqlen_q, H, D]

    return torch.cat(outputs, dim=0)  # [total_tokens, num_qo_heads, head_dim]


class PyTorchBackend(FlashAttentionBackend):
    """Paged attention backend built on PyTorch SDPA — no custom kernels."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, FAMetadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        return _pt_paged_attn(
            q=q,
            k_cache=self.kvcache.k_cache(layer_id),
            v_cache=self.kvcache.v_cache(layer_id),
            page_table=metadata.page_table,
            cache_seqlens=metadata.cache_seqlens,
            cu_seqlens_q=metadata.cu_seqlens_q,
            page_size=self.page_size,
            softmax_scale=self.scale,
        )
