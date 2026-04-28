"""Triton RMSNorm kernels — drop-in replacements for `flashinfer.rmsnorm`
and `flashinfer.fused_add_rmsnorm` so the inference engine can run on ROCm.

Both kernels keep the flashinfer call signatures byte-for-byte:

    triton_rmsnorm(x, weight, eps, out=None)        -> Tensor
    triton_fused_add_rmsnorm(x, residual, weight, eps) -> None  (in-place)

The fused variant updates `residual` to `residual + x` and overwrites `x`
with the normalized result in a single pass — that single-pass property is
the whole reason the fused op exists, so we keep it.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    stride_x_row,
    stride_out_row,
    n_cols: tl.constexpr,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_x_row
    out_row = out_ptr + row * stride_out_row

    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(out_row + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _fused_add_rmsnorm_kernel(
    x_ptr,
    residual_ptr,
    w_ptr,
    stride_x_row,
    stride_r_row,
    n_cols: tl.constexpr,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_x_row
    r_row = residual_ptr + row * stride_r_row

    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_row + cols, mask=mask, other=0.0).to(tl.float32)
    r_new = r + x
    tl.store(r_row + cols, r_new.to(residual_ptr.dtype.element_ty), mask=mask)

    var = tl.sum(r_new * r_new, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = r_new * rstd * w
    tl.store(x_row + cols, y.to(x_ptr.dtype.element_ty), mask=mask)


_MAX_SUPPORTED_HIDDEN = 32768


def _pick_num_warps(block: int) -> int:
    if block <= 1024:
        return 4
    if block <= 4096:
        return 8
    return 16


def _flatten2d(t: torch.Tensor) -> torch.Tensor:
    n = t.shape[-1]
    return t.reshape(-1, n)


def triton_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match `flashinfer.rmsnorm`: out = x * rsqrt(mean(x^2) + eps) * weight (along last dim)."""
    assert x.is_cuda and weight.is_cuda, "rmsnorm inputs must be on GPU"
    assert weight.ndim == 1 and weight.shape[0] == x.shape[-1]
    assert x.is_contiguous(), "x must be contiguous"
    assert weight.is_contiguous(), "weight must be contiguous"

    n = x.shape[-1]
    assert n <= _MAX_SUPPORTED_HIDDEN, (
        f"hidden_dim={n} exceeds single-block RMSNorm capacity "
        f"({_MAX_SUPPORTED_HIDDEN}); split-K variant not implemented yet"
    )

    if out is None:
        out = torch.empty_like(x)
    else:
        assert out.shape == x.shape and out.dtype == x.dtype
        assert out.is_contiguous()

    x2 = _flatten2d(x)
    out2 = _flatten2d(out)
    M = x2.shape[0]
    if M == 0:
        return out
    BLOCK = triton.next_power_of_2(n)
    _rmsnorm_kernel[(M,)](
        x2,
        weight,
        out2,
        x2.stride(0),
        out2.stride(0),
        n,
        float(eps),
        BLOCK=BLOCK,
        num_warps=_pick_num_warps(BLOCK),
    )
    return out


def triton_fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> None:
    """Match `flashinfer.fused_add_rmsnorm` — in-place:
        residual <- residual + x
        x        <- rmsnorm(residual_new) * weight
    """
    assert x.is_cuda and residual.is_cuda and weight.is_cuda
    assert x.shape == residual.shape and x.dtype == residual.dtype
    assert weight.ndim == 1 and weight.shape[0] == x.shape[-1]
    assert x.is_contiguous() and residual.is_contiguous() and weight.is_contiguous()

    n = x.shape[-1]
    assert n <= _MAX_SUPPORTED_HIDDEN, (
        f"hidden_dim={n} exceeds single-block RMSNorm capacity "
        f"({_MAX_SUPPORTED_HIDDEN}); split-K variant not implemented yet"
    )

    x2 = _flatten2d(x)
    r2 = _flatten2d(residual)
    M = x2.shape[0]
    if M == 0:
        return
    BLOCK = triton.next_power_of_2(n)
    _fused_add_rmsnorm_kernel[(M,)](
        x2,
        r2,
        weight,
        x2.stride(0),
        r2.stride(0),
        n,
        float(eps),
        BLOCK=BLOCK,
        num_warps=_pick_num_warps(BLOCK),
    )
