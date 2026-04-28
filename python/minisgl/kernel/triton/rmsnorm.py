"""Triton RMSNorm kernels — drop-in replacements for `flashinfer.rmsnorm`
and `flashinfer.fused_add_rmsnorm` so the inference engine can run on ROCm.

Both kernels keep the flashinfer call signatures byte-for-byte:

    triton_rmsnorm(x, weight, eps, out=None)        -> Tensor
    triton_fused_add_rmsnorm(x, residual, weight, eps) -> None  (in-place)

The fused variant updates `residual` to `residual + x` and overwrites `x`
with the normalized result in a single pass — that single-pass property is
the whole reason the fused op exists, so we keep it.

Inputs may be 2D `[M, N]` or 3D `[D0, D1, N]`; only the **last** dim must be
contiguous (stride 1). Outer dims may be strided (e.g. `qkv.split(...)` slices
viewed as 3D). Higher rank is collapsed to a contiguous `[M, N]` first.
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
    stride_x_outer,
    stride_x_inner,
    stride_o_outer,
    stride_o_inner,
    n_cols: tl.constexpr,
    eps,
    BLOCK: tl.constexpr,
):
    pid_outer = tl.program_id(0)
    pid_inner = tl.program_id(1)
    x_row = x_ptr + pid_outer * stride_x_outer + pid_inner * stride_x_inner
    out_row = out_ptr + pid_outer * stride_o_outer + pid_inner * stride_o_inner

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
    stride_x_outer,
    stride_x_inner,
    stride_r_outer,
    stride_r_inner,
    n_cols: tl.constexpr,
    eps,
    BLOCK: tl.constexpr,
):
    pid_outer = tl.program_id(0)
    pid_inner = tl.program_id(1)
    x_row = x_ptr + pid_outer * stride_x_outer + pid_inner * stride_x_inner
    r_row = residual_ptr + pid_outer * stride_r_outer + pid_inner * stride_r_inner

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


def _grid_for(t: torch.Tensor):
    """Return (outer, inner, stride_outer, stride_inner) for a 2D or 3D tensor whose
    last dim is contiguous. Higher rank must be collapsed by the caller first."""
    if t.ndim == 1:
        return 1, 1, 0, 0
    if t.ndim == 2:
        return t.shape[0], 1, t.stride(0), 0
    assert t.ndim == 3, f"unexpected rank {t.ndim} after collapse"
    return t.shape[0], t.shape[1], t.stride(0), t.stride(1)


def _normalize_input(t: torch.Tensor, n: int) -> torch.Tensor:
    """Collapse to a layout the kernel can address: ndim<=3, last-dim stride 1."""
    assert t.shape[-1] == n
    if t.ndim <= 3 and t.stride(-1) == 1:
        return t
    # Higher rank or strided last-dim → make contiguous and flatten leading dims.
    return t.contiguous().view(-1, n)


def triton_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match `flashinfer.rmsnorm`: out = x * rsqrt(mean(x^2) + eps) * weight (along last dim)."""
    assert x.is_cuda and weight.is_cuda
    assert weight.ndim == 1 and weight.shape[0] == x.shape[-1]
    assert weight.is_contiguous()

    n = x.shape[-1]
    assert n <= _MAX_SUPPORTED_HIDDEN, (
        f"hidden_dim={n} exceeds single-block RMSNorm capacity ({_MAX_SUPPORTED_HIDDEN})"
    )
    inplace = out is x
    if out is None:
        out = torch.empty_like(x)
    elif not inplace:
        assert out.shape == x.shape and out.dtype == x.dtype

    x_v = _normalize_input(x, n)
    out_v = out if inplace else _normalize_input(out, n)
    if x_v.numel() == 0:
        return out
    assert x_v.stride(-1) == 1 and out_v.stride(-1) == 1

    outer, inner, sx_o, sx_i = _grid_for(x_v)
    _, _, so_o, so_i = _grid_for(out_v)

    BLOCK = triton.next_power_of_2(n)
    _rmsnorm_kernel[(outer, inner)](
        x_v,
        weight,
        out_v,
        sx_o,
        sx_i,
        so_o,
        so_i,
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
    assert weight.is_contiguous()

    n = x.shape[-1]
    assert n <= _MAX_SUPPORTED_HIDDEN, (
        f"hidden_dim={n} exceeds single-block RMSNorm capacity ({_MAX_SUPPORTED_HIDDEN})"
    )

    x_v = _normalize_input(x, n)
    r_v = _normalize_input(residual, n)
    if x_v.numel() == 0:
        return
    assert x_v.stride(-1) == 1 and r_v.stride(-1) == 1

    outer, inner, sx_o, sx_i = _grid_for(x_v)
    _, _, sr_o, sr_i = _grid_for(r_v)

    BLOCK = triton.next_power_of_2(n)
    _fused_add_rmsnorm_kernel[(outer, inner)](
        x_v,
        r_v,
        weight,
        sx_o,
        sx_i,
        sr_o,
        sr_i,
        n,
        float(eps),
        BLOCK=BLOCK,
        num_warps=_pick_num_warps(BLOCK),
    )
