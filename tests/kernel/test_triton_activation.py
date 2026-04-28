"""Numerical correctness for the Triton silu_and_mul / gelu_and_mul kernels."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from minisgl.kernel.triton.activation import (
    triton_gelu_and_mul,
    triton_silu_and_mul,
)


def _require_gpu():
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")


def _ref_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    D = x.shape[-1] // 2
    gate, up = x[..., :D], x[..., D:]
    return F.silu(gate.float()).to(x.dtype) * up


def _ref_gelu_and_mul(x: torch.Tensor) -> torch.Tensor:
    D = x.shape[-1] // 2
    gate, up = x[..., :D], x[..., D:]
    return F.gelu(gate.float(), approximate="tanh").to(x.dtype) * up


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("D", [128, 1024, 4096])
@pytest.mark.parametrize("rows", [1, 7, 128])
def test_silu_and_mul_matches_eager(dtype, D, rows):
    _require_gpu()
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    x = torch.randn(rows, 2 * D, dtype=dtype, device=device)
    out = triton_silu_and_mul(x)
    ref = _ref_silu_and_mul(x)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("D", [256, 2048])
def test_gelu_and_mul_matches_eager(D):
    _require_gpu()
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    x = torch.randn(32, 2 * D, dtype=torch.float16, device=device)
    out = triton_gelu_and_mul(x)
    ref = _ref_gelu_and_mul(x)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


def test_silu_and_mul_inplace_via_out():
    _require_gpu()
    device = torch.device("cuda:0")
    x = torch.randn(16, 1024, dtype=torch.float16, device=device)
    out = torch.empty(16, 512, dtype=torch.float16, device=device)
    ref = _ref_silu_and_mul(x)
    triton_silu_and_mul(x, out=out)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


def test_silu_and_mul_higher_rank():
    _require_gpu()
    device = torch.device("cuda:0")
    x = torch.randn(2, 16, 512, dtype=torch.float16, device=device)
    out = triton_silu_and_mul(x)
    assert out.shape == (2, 16, 256)
    torch.testing.assert_close(out, _ref_silu_and_mul(x), atol=2e-2, rtol=2e-2)
