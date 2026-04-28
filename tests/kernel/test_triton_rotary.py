"""Numerical correctness for the Triton RoPE kernel."""
from __future__ import annotations

import pytest
import torch

from minisgl.kernel.triton.rotary import (
    triton_apply_rope_with_cos_sin_cache_inplace,
)


def _require_gpu():
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")


def _build_cos_sin_cache(max_pos: int, head_size: int, base: float, device, dtype):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_size, 2, dtype=torch.float32, device=device) / head_size))
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # [max_pos, head_size/2]
    cos = freqs.cos()
    sin = freqs.sin()
    return torch.cat((cos, sin), dim=-1).to(dtype)  # [max_pos, head_size]


def _eager_rope(x: torch.Tensor, positions: torch.Tensor, cs: torch.Tensor, head_size: int) -> torch.Tensor:
    """GPT-NeoX split-half RoPE reference. x: [..., H*head_size]."""
    half = head_size // 2
    T = x.shape[0]
    H = x.shape[-1] // head_size
    x3 = x.view(T, H, head_size).clone().float()
    a = x3[..., :half]
    b = x3[..., half:]

    cs_rows = cs[positions.long()].float()  # [T, head_size]
    cos = cs_rows[..., :half].unsqueeze(1)  # [T, 1, half]
    sin = cs_rows[..., half:].unsqueeze(1)

    a_new = a * cos - b * sin
    b_new = a * sin + b * cos
    out = torch.cat([a_new, b_new], dim=-1)
    return out.view(T, H * head_size).to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_size", [64, 128, 256])
@pytest.mark.parametrize("num_qo,num_kv", [(8, 8), (8, 1), (32, 4)])
def test_rope_matches_eager(dtype, head_size, num_qo, num_kv):
    _require_gpu()
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    T = 17
    max_pos = 4096

    cs = _build_cos_sin_cache(max_pos, head_size, base=10000.0, device=device, dtype=dtype)
    positions = torch.randint(0, max_pos, (T,), dtype=torch.int32, device=device)

    q = torch.randn(T, num_qo * head_size, dtype=dtype, device=device)
    k = torch.randn(T, num_kv * head_size, dtype=dtype, device=device)

    expected_q = _eager_rope(q, positions, cs, head_size)
    expected_k = _eager_rope(k, positions, cs, head_size)

    triton_apply_rope_with_cos_sin_cache_inplace(
        positions=positions, query=q, key=k, head_size=head_size, cos_sin_cache=cs,
    )

    torch.testing.assert_close(q, expected_q, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k, expected_k, atol=2e-2, rtol=2e-2)


def test_rope_handles_split_qkv_view():
    """Q and K are typically slices of a wider qkv tensor — strides aren't `head_size*H`.
    Verify the kernel respects per-tensor strides."""
    _require_gpu()
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    T, head_size, num_qo, num_kv = 13, 128, 8, 2
    max_pos = 1024

    cs = _build_cos_sin_cache(max_pos, head_size, base=10000.0, device=device, dtype=torch.float16)
    positions = torch.arange(T, dtype=torch.int32, device=device)

    qo_dim = num_qo * head_size
    kv_dim = num_kv * head_size
    qkv = torch.randn(T, qo_dim + 2 * kv_dim, dtype=torch.float16, device=device)
    q, k, v = qkv.split([qo_dim, kv_dim, kv_dim], dim=-1)

    q_ref = _eager_rope(q.contiguous(), positions, cs, head_size)
    k_ref = _eager_rope(k.contiguous(), positions, cs, head_size)
    v_before = v.clone()

    triton_apply_rope_with_cos_sin_cache_inplace(
        positions=positions, query=q, key=k, head_size=head_size, cos_sin_cache=cs,
    )

    torch.testing.assert_close(q, q_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k, k_ref, atol=2e-2, rtol=2e-2)
    # V must NOT have been touched
    torch.testing.assert_close(v, v_before)


def test_rope_int64_positions():
    """Positions tensor in int64 should also work."""
    _require_gpu()
    device = torch.device("cuda:0")
    T, head_size, num_qo, num_kv = 5, 64, 4, 4
    cs = _build_cos_sin_cache(256, head_size, 10000.0, device, torch.float16)
    positions = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=device)
    q = torch.randn(T, num_qo * head_size, dtype=torch.float16, device=device)
    k = torch.randn(T, num_kv * head_size, dtype=torch.float16, device=device)

    expected_q = _eager_rope(q, positions, cs, head_size)
    expected_k = _eager_rope(k, positions, cs, head_size)

    triton_apply_rope_with_cos_sin_cache_inplace(
        positions=positions, query=q, key=k, head_size=head_size, cos_sin_cache=cs,
    )

    torch.testing.assert_close(q, expected_q, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k, expected_k, atol=2e-2, rtol=2e-2)


def test_rope_empty_input_is_noop():
    _require_gpu()
    device = torch.device("cuda:0")
    cs = _build_cos_sin_cache(64, 64, 10000.0, device, torch.float16)
    positions = torch.empty(0, dtype=torch.int32, device=device)
    q = torch.empty(0, 4 * 64, dtype=torch.float16, device=device)
    k = torch.empty(0, 4 * 64, dtype=torch.float16, device=device)
    # Must not raise or launch a kernel with grid=0.
    triton_apply_rope_with_cos_sin_cache_inplace(
        positions=positions, query=q, key=k, head_size=64, cos_sin_cache=cs,
    )
