from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    pass


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    half = x.shape[-1] // 2
    result = F.silu(x[..., :half]) * x[..., half:]
    if out is not None:
        out.copy_(result)
        return out
    return result


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    half = x.shape[-1] // 2
    result = F.gelu(x[..., :half]) * x[..., half:]
    if out is not None:
        out.copy_(result)
        return out
    return result


__all__ = ["silu_and_mul", "gelu_and_mul"]
