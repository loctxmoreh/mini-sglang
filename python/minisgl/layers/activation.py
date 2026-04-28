from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Callable

from minisgl.utils import is_fi_available

if TYPE_CHECKING:
    import torch


@functools.cache
def _get_silu_and_mul() -> Callable:
    if is_fi_available():
        from flashinfer import silu_and_mul as fi_silu_and_mul

        return fi_silu_and_mul
    from minisgl.kernel.triton.activation import triton_silu_and_mul

    return triton_silu_and_mul


@functools.cache
def _get_gelu_and_mul() -> Callable:
    if is_fi_available():
        from flashinfer import gelu_and_mul as fi_gelu_and_mul

        return fi_gelu_and_mul
    from minisgl.kernel.triton.activation import triton_gelu_and_mul

    return triton_gelu_and_mul


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    return _get_silu_and_mul()(x, out=out)


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    return _get_gelu_and_mul()(x, out=out)


__all__ = ["silu_and_mul", "gelu_and_mul"]
