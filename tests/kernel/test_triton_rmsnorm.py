"""Numerical correctness for the Triton RMSNorm kernels."""
from __future__ import annotations

import pytest
import torch

from minisgl.kernel.triton.rmsnorm import (
    triton_fused_add_rmsnorm,
    triton_rmsnorm,
)


def _require_gpu():
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")


def _ref_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x * torch.rsqrt(var + eps).to(x.dtype)) * weight


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden", [128, 1024, 4096])
@pytest.mark.parametrize("rows", [1, 7, 128])
def test_rmsnorm_matches_eager(dtype, hidden, rows):
    _require_gpu()
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    x = torch.randn(rows, hidden, dtype=dtype, device=device)
    w = torch.randn(hidden, dtype=dtype, device=device)
    eps = 1e-6

    out = triton_rmsnorm(x, w, eps)
    ref = _ref_rmsnorm(x, w, eps)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


def test_rmsnorm_inplace():
    _require_gpu()
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    x = torch.randn(8, 1024, dtype=torch.float16, device=device)
    w = torch.randn(1024, dtype=torch.float16, device=device)
    ref = _ref_rmsnorm(x, w, 1e-6)
    triton_rmsnorm(x, w, 1e-6, out=x)
    torch.testing.assert_close(x, ref, atol=2e-2, rtol=2e-2)


def test_rmsnorm_higher_rank_input():
    """Input of shape [B, T, H] should normalize along the last dim."""
    _require_gpu()
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    x = torch.randn(2, 16, 512, dtype=torch.float16, device=device)
    w = torch.randn(512, dtype=torch.float16, device=device)
    out = triton_rmsnorm(x, w, 1e-6)
    ref = _ref_rmsnorm(x, w, 1e-6)
    assert out.shape == (2, 16, 512)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden", [256, 2048, 4096])
def test_fused_add_rmsnorm_matches_eager(dtype, hidden):
    _require_gpu()
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    rows = 32
    x = torch.randn(rows, hidden, dtype=dtype, device=device)
    residual = torch.randn(rows, hidden, dtype=dtype, device=device)
    w = torch.randn(hidden, dtype=dtype, device=device)
    eps = 1e-6

    # eager reference: residual_new = residual + x; x_new = rmsnorm(residual_new)
    expected_residual = residual + x
    expected_x = _ref_rmsnorm(expected_residual, w, eps)

    triton_fused_add_rmsnorm(x, residual, w, eps)

    torch.testing.assert_close(residual, expected_residual, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(x, expected_x, atol=2e-2, rtol=2e-2)


def test_fused_add_rmsnorm_does_not_clobber_unrelated_rows():
    """Each row's update must not bleed into neighbors (single-pass single-row kernel)."""
    _require_gpu()
    torch.manual_seed(1)
    device = torch.device("cuda:0")
    rows, h = 4, 512
    x_orig = torch.randn(rows, h, dtype=torch.float16, device=device)
    r_orig = torch.randn(rows, h, dtype=torch.float16, device=device)
    w = torch.randn(h, dtype=torch.float16, device=device)
    x = x_orig.clone()
    r = r_orig.clone()

    triton_fused_add_rmsnorm(x, r, w, 1e-6)

    # Manually recompute row 2 as a sanity check
    expected_r2 = r_orig[2] + x_orig[2]
    expected_x2 = _ref_rmsnorm(expected_r2, w, 1e-6)
    torch.testing.assert_close(r[2], expected_r2, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(x[2], expected_x2, atol=2e-2, rtol=2e-2)
