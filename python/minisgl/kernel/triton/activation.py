"""Triton fused activation-and-multiply kernels — replacements for
`flashinfer.silu_and_mul` / `flashinfer.gelu_and_mul`.

Input `x` has shape `[..., 2*D]` and is split into the two halves along the last
dim: `gate = x[..., :D]`, `up = x[..., D:]`. Output is `act(gate) * up` with
shape `[..., D]`.

Last dim of `x` and `out` must be unit-strided; outer dims may be strided.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _act_and_mul_kernel(
    x_ptr,
    out_ptr,
    stride_x_outer,
    stride_x_inner,
    stride_o_outer,
    stride_o_inner,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    ACT: tl.constexpr,  # 0 = silu, 1 = gelu (tanh approx)
):
    pid_outer = tl.program_id(0)
    pid_inner = tl.program_id(1)
    x_row = x_ptr + pid_outer * stride_x_outer + pid_inner * stride_x_inner
    o_row = out_ptr + pid_outer * stride_o_outer + pid_inner * stride_o_inner

    cols = tl.arange(0, BLOCK)
    mask = cols < D
    gate = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_row + D + cols, mask=mask, other=0.0).to(tl.float32)

    if ACT == 0:
        activated = gate * (1.0 / (1.0 + tl.exp(-gate)))
    else:
        # GELU tanh approximation: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
        kAlpha = 0.7978845608028654  # sqrt(2/pi)
        kBeta = 0.044715
        inner = kAlpha * (gate + kBeta * gate * gate * gate)
        # tanh(x) = 1 - 2 / (exp(2x) + 1) — Triton 3.1 has no tl.tanh on ROCm
        tanh_inner = 1.0 - 2.0 / (tl.exp(2.0 * inner) + 1.0)
        activated = 0.5 * gate * (1.0 + tanh_inner)

    y = activated * up
    tl.store(o_row + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


def _grid_for(t: torch.Tensor):
    if t.ndim == 1:
        return 1, 1, 0, 0
    if t.ndim == 2:
        return t.shape[0], 1, t.stride(0), 0
    assert t.ndim == 3, f"unexpected rank {t.ndim} after collapse"
    return t.shape[0], t.shape[1], t.stride(0), t.stride(1)


def _normalize(t: torch.Tensor, last: int) -> torch.Tensor:
    assert t.shape[-1] == last
    if t.ndim <= 3 and t.stride(-1) == 1:
        return t
    return t.contiguous().view(-1, last)


def _act_and_mul(x: torch.Tensor, out: torch.Tensor | None, act: int) -> torch.Tensor:
    assert x.is_cuda
    assert x.shape[-1] % 2 == 0
    D = x.shape[-1] // 2
    out_shape = x.shape[:-1] + (D,)

    if out is None:
        out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    else:
        assert out.shape == out_shape and out.dtype == x.dtype

    x_v = _normalize(x, 2 * D)
    out_v = _normalize(out, D)
    if x_v.numel() == 0:
        return out
    assert x_v.stride(-1) == 1 and out_v.stride(-1) == 1

    outer, inner, sx_o, sx_i = _grid_for(x_v)
    _, _, so_o, so_i = _grid_for(out_v)

    BLOCK = triton.next_power_of_2(D)
    num_warps = 4 if BLOCK <= 1024 else (8 if BLOCK <= 4096 else 16)
    _act_and_mul_kernel[(outer, inner)](
        x_v,
        out_v,
        sx_o,
        sx_i,
        so_o,
        so_i,
        D,
        BLOCK=BLOCK,
        ACT=act,
        num_warps=num_warps,
    )
    return out


def triton_silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Match `flashinfer.silu_and_mul`."""
    return _act_and_mul(x, out, act=0)


def triton_gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Match `flashinfer.gelu_and_mul` (tanh approximation)."""
    return _act_and_mul(x, out, act=1)
