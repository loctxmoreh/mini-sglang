"""Triton sampler — drop-in for `flashinfer.sampling.{softmax, *_from_probs}`.

We do the heavy single-pass per-row work in a Triton kernel (tempered softmax,
fused with max + sum + write-back) and lean on `torch.cumsum`/`torch.searchsorted`
/`torch.topk`/`torch.sort` for the sampling/top-k/top-p stages, since those are
already well-tuned in PyTorch's ROCm build and don't benefit from a hand-rolled
Triton kernel at vocab sizes ≤ a few hundred thousand.
"""
from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: tempered batched softmax over rows of [B, V]
# ---------------------------------------------------------------------------

@triton.jit
def _softmax_temp_kernel(
    logits_ptr,
    temps_ptr,
    probs_ptr,
    stride_lb,
    stride_pb,
    V,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    inv_t = 1.0 / tl.load(temps_ptr + pid).to(tl.float32)
    logits_row = logits_ptr + pid * stride_lb
    probs_row = probs_ptr + pid * stride_pb

    # Pass 1: online max + sum.
    m = -float("inf")
    s = 0.0
    for i in range(0, V, BLOCK):
        cols = i + tl.arange(0, BLOCK)
        mask = cols < V
        x = tl.load(logits_row + cols, mask=mask, other=-float("inf")).to(tl.float32)
        x_t = x * inv_t
        m_block = tl.max(x_t, axis=0)
        m_new = tl.maximum(m, m_block)
        e = tl.where(mask, tl.exp(x_t - m_new), 0.0)
        s = s * tl.exp(m - m_new) + tl.sum(e, axis=0)
        m = m_new

    inv_s = 1.0 / s

    # Pass 2: write probs.
    for i in range(0, V, BLOCK):
        cols = i + tl.arange(0, BLOCK)
        mask = cols < V
        x = tl.load(logits_row + cols, mask=mask, other=-float("inf")).to(tl.float32)
        p = tl.exp(x * inv_t - m) * inv_s
        p = tl.where(mask, p, 0.0)
        tl.store(probs_row + cols, p.to(probs_ptr.dtype.element_ty), mask=mask)


def triton_softmax(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    enable_pdl: bool = False,  # accepted for API parity; ignored
) -> torch.Tensor:
    """Match `flashinfer.sampling.softmax(logits, temperatures)`.

    `logits` is `[B, V]`, `temperatures` is `[B]`; returns `[B, V]` probs.
    """
    assert logits.is_cuda and temperatures.is_cuda
    assert logits.ndim == 2
    assert temperatures.shape == (logits.shape[0],)
    assert logits.stride(-1) == 1, "logits must be contiguous on last dim"

    B, V = logits.shape
    probs = torch.empty_like(logits)
    if B == 0:
        return probs

    BLOCK = min(1 << 12, triton.next_power_of_2(V))  # cap block at 4K to bound register pressure
    num_warps = 4 if BLOCK <= 1024 else 8
    _softmax_temp_kernel[(B,)](
        logits,
        temperatures.contiguous(),
        probs,
        logits.stride(0),
        probs.stride(0),
        V,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return probs


# ---------------------------------------------------------------------------
# Sampling helpers — torch-native, exploiting per-row independence.
# ---------------------------------------------------------------------------

def _categorical_from_probs(
    probs: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample one index per row by inverse-CDF.

    `probs` rows need not be normalized — we scale the uniform by the row's total
    mass, which makes top-k/top-p (mass-removed) inputs work without explicit
    renormalization.
    """
    cdf = torch.cumsum(probs, dim=-1)
    last = cdf[..., -1:].clamp_min(1e-30)
    u = torch.rand(
        probs.shape[:-1] + (1,),
        device=probs.device,
        dtype=cdf.dtype,
        generator=generator,
    ) * last
    idx = torch.searchsorted(cdf, u).squeeze(-1)
    return idx.clamp_(max=probs.shape[-1] - 1).to(torch.int32)


def triton_sampling_from_probs(
    probs: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Match `flashinfer.sampling.sampling_from_probs`."""
    return _categorical_from_probs(probs, generator=generator)


def _topk_mask(probs: torch.Tensor, top_k: torch.Tensor | int) -> torch.Tensor:
    """Return a copy of `probs` with everything outside the per-row top-k set to 0."""
    V = probs.shape[-1]
    if isinstance(top_k, torch.Tensor):
        max_k = int(top_k.max().item())
    else:
        max_k = int(top_k)
    max_k = min(max_k, V)
    if max_k <= 0:
        return torch.zeros_like(probs)

    vals, idx = torch.topk(probs, k=max_k, dim=-1)
    if isinstance(top_k, torch.Tensor):
        col_range = torch.arange(max_k, device=probs.device).unsqueeze(0)  # [1, max_k]
        active = col_range < top_k.to(col_range.dtype).unsqueeze(-1)        # [B, max_k]
        vals = vals * active

    out = torch.zeros_like(probs)
    out.scatter_(-1, idx, vals)
    return out


def _topp_mask_descending(
    sorted_probs: torch.Tensor, top_p: torch.Tensor | float
) -> torch.Tensor:
    """Apply top-p (nucleus) on already-sorted-descending probs.

    Keeps the smallest prefix whose cumulative mass exceeds `top_p`. The boundary
    token IS included; the first token is always kept (matches the SGLang/HF
    nucleus-sampling convention).
    """
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    cumsum_before = cumsum - sorted_probs
    if isinstance(top_p, torch.Tensor):
        thresh = top_p.to(sorted_probs.dtype).unsqueeze(-1)
    else:
        thresh = float(top_p)
    keep = cumsum_before < thresh
    keep[..., 0] = True
    return sorted_probs * keep


def triton_top_k_sampling_from_probs(
    probs: torch.Tensor,
    top_k: torch.Tensor | int,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Match `flashinfer.sampling.top_k_sampling_from_probs`."""
    masked = _topk_mask(probs, top_k)
    return _categorical_from_probs(masked, generator=generator)


def triton_top_p_sampling_from_probs(
    probs: torch.Tensor,
    top_p: torch.Tensor | float,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Match `flashinfer.sampling.top_p_sampling_from_probs`."""
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    masked = _topp_mask_descending(sorted_probs, top_p)
    local = _categorical_from_probs(masked, generator=generator)
    return sorted_idx.gather(-1, local.unsqueeze(-1).to(torch.int64)).squeeze(-1).to(torch.int32)


def triton_top_k_top_p_sampling_from_probs(
    probs: torch.Tensor,
    top_k: torch.Tensor | int,
    top_p: torch.Tensor | float,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Match `flashinfer.sampling.top_k_top_p_sampling_from_probs`.

    Top-k first (cheap), then top-p on the resulting `[B, max_k]` slab; finally
    map the local index back through the top-k indices.
    """
    V = probs.shape[-1]
    if isinstance(top_k, torch.Tensor):
        max_k = int(top_k.max().item())
    else:
        max_k = int(top_k)
    max_k = min(max_k, V)

    vals, idx = torch.topk(probs, k=max_k, dim=-1)  # already sorted descending
    if isinstance(top_k, torch.Tensor):
        col_range = torch.arange(max_k, device=probs.device).unsqueeze(0)
        active = col_range < top_k.to(col_range.dtype).unsqueeze(-1)
        vals = vals * active

    masked = _topp_mask_descending(vals, top_p)
    local = _categorical_from_probs(masked, generator=generator)
    return idx.gather(-1, local.unsqueeze(-1).to(torch.int64)).squeeze(-1).to(torch.int32)
