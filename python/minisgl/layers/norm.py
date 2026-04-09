from typing import Tuple

import torch
import triton
import triton.language as tl

from .base import BaseOP


@triton.jit
def _rmsnorm_kernel(
    X,           # pointer: [M, N]
    W,           # pointer: [N]
    Y,           # pointer: [M, N]
    stride_xm,
    N,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    x_raw = tl.load(X + row * stride_xm + offs, mask=mask, other=0.0)
    x_f32 = x_raw.to(tl.float32)

    var = tl.sum(x_f32 * x_f32) / N
    rrms = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W + offs, mask=mask, other=1.0).to(tl.float32)
    y = (x_f32 * rrms * w).to(x_raw.dtype)
    tl.store(Y + row * stride_xm + offs, y, mask=mask)


@triton.jit
def _fused_add_rmsnorm_kernel(
    X,           # pointer: [M, N]  — overwritten with rmsnorm result
    Residual,    # pointer: [M, N]  — overwritten with residual + x
    W,           # pointer: [N]
    stride_m,
    N,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    x_raw = tl.load(X + row * stride_m + offs, mask=mask, other=0.0)
    r_raw = tl.load(Residual + row * stride_m + offs, mask=mask, other=0.0)

    x_f32 = x_raw.to(tl.float32)
    r_f32 = r_raw.to(tl.float32)

    new_r = r_f32 + x_f32
    tl.store(Residual + row * stride_m + offs, new_r.to(r_raw.dtype), mask=mask)

    var = tl.sum(new_r * new_r) / N
    rrms = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W + offs, mask=mask, other=1.0).to(tl.float32)
    y = (new_r * rrms * w).to(x_raw.dtype)
    tl.store(X + row * stride_m + offs, y, mask=mask)


def _triton_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    shape = x.shape
    x_2d = x.view(-1, shape[-1])
    M, N = x_2d.shape
    BLOCK_N = triton.next_power_of_2(N)
    y_2d = out.view(-1, shape[-1]) if out is not None else torch.empty_like(x_2d)
    _rmsnorm_kernel[(M,)](x_2d, weight, y_2d, x_2d.stride(0), N, eps, BLOCK_N=BLOCK_N)
    return out if out is not None else y_2d.view(shape)


def _triton_fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> None:
    """In-place: residual += x; x = rmsnorm(residual)."""
    shape = x.shape
    x_2d = x.view(-1, shape[-1])
    r_2d = residual.view(-1, shape[-1])
    M, N = x_2d.shape
    BLOCK_N = triton.next_power_of_2(N)
    _fused_add_rmsnorm_kernel[(M,)](
        x_2d, r_2d, weight, x_2d.stride(0), N, eps, BLOCK_N=BLOCK_N
    )


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _triton_rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        _triton_rmsnorm(x, self.weight, self.eps, out=x)


class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return _triton_rmsnorm(x, self.weight, self.eps), x
        _triton_fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual
