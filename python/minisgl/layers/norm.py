from __future__ import annotations

import functools
import importlib.util
from typing import Callable, Tuple

import torch

from .base import BaseOP


@functools.cache
def is_fi_available() -> bool:
    """Whether `flashinfer` is importable in this environment."""
    return importlib.util.find_spec("flashinfer") is not None


@functools.cache
def _get_rmsnorm_impls() -> Tuple[Callable, Callable]:
    """Return `(rmsnorm, fused_add_rmsnorm)` — flashinfer if available, else Triton."""
    if is_fi_available():
        from flashinfer import fused_add_rmsnorm, rmsnorm

        return rmsnorm, fused_add_rmsnorm
    from minisgl.kernel.triton.rmsnorm import triton_fused_add_rmsnorm, triton_rmsnorm

    return triton_rmsnorm, triton_fused_add_rmsnorm


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm, _ = _get_rmsnorm_impls()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        self.rmsnorm(x, self.weight, self.eps, out=x)


class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm, self.fused_add_rmsnorm = _get_rmsnorm_impls()

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rmsnorm(x, self.weight, self.eps), x
        self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual
