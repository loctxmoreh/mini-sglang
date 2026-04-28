"""Triton RoPE kernel — drop-in replacement for
`flashinfer.apply_rope_with_cos_sin_cache_inplace`.

Signature (matches flashinfer exactly):

    triton_apply_rope_with_cos_sin_cache_inplace(
        positions, query, key, head_size, cos_sin_cache,
    ) -> None  (in-place on query and key)

Convention is GPT-NeoX / split-half RoPE:
  for j in [0, head_size/2):
    a = x[..., j]
    b = x[..., j + head_size/2]
    x[..., j]               = a * cos[pos, j] - b * sin[pos, j]
    x[..., j + head_size/2] = a * sin[pos, j] + b * cos[pos, j]

`cos_sin_cache` is laid out as `[max_pos, head_size]`, where each row is
`cat(cos[pos, :head_size/2], sin[pos, :head_size/2])` — same layout that
`RotaryEmbedding.__init__` already builds.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_inplace_kernel(
    x_ptr,
    pos_ptr,
    cs_ptr,
    stride_x_t,
    stride_x_h,
    stride_cs,
    HEAD_SIZE: tl.constexpr,
    HALF: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    pos = tl.load(pos_ptr + pid_t).to(tl.int64)
    cs_row = cs_ptr + pos * stride_cs

    offs = tl.arange(0, HALF)
    cos = tl.load(cs_row + offs).to(tl.float32)
    sin = tl.load(cs_row + HALF + offs).to(tl.float32)

    base = x_ptr + pid_t * stride_x_t + pid_h * stride_x_h
    a = tl.load(base + offs).to(tl.float32)
    b = tl.load(base + HALF + offs).to(tl.float32)

    a_new = a * cos - b * sin
    b_new = a * sin + b * cos

    out_dtype = x_ptr.dtype.element_ty
    tl.store(base + offs, a_new.to(out_dtype))
    tl.store(base + HALF + offs, b_new.to(out_dtype))


def _rotate_inplace(x: torch.Tensor, positions: torch.Tensor, cs: torch.Tensor, head_size: int):
    """Rotate one tensor `x` of shape [T, H, head_size] (or [T, H*head_size]) in place."""
    T = x.shape[0]
    flat_dim = x.shape[-1] if x.ndim == 3 else x.shape[-1]
    if x.ndim == 2:
        assert x.shape[1] % head_size == 0
        H = x.shape[1] // head_size
        x3 = x.view(T, H, head_size)
    else:
        assert x.ndim == 3 and x.shape[2] == head_size
        H = x.shape[1]
        x3 = x

    half = head_size // 2
    grid = (T, H)
    _rope_inplace_kernel[grid](
        x3,
        positions,
        cs,
        x3.stride(0),
        x3.stride(1),
        cs.stride(0),
        HEAD_SIZE=head_size,
        HALF=half,
        num_warps=4,
    )


def triton_apply_rope_with_cos_sin_cache_inplace(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
) -> None:
    """Match `flashinfer.apply_rope_with_cos_sin_cache_inplace` byte-for-byte."""
    assert positions.is_cuda and query.is_cuda and key.is_cuda and cos_sin_cache.is_cuda
    assert positions.dtype in (torch.int32, torch.int64), positions.dtype
    assert head_size in (64, 128, 256, 512), head_size
    assert head_size % 2 == 0
    assert cos_sin_cache.ndim == 2 and cos_sin_cache.shape[1] == head_size, cos_sin_cache.shape
    assert query.shape[0] == key.shape[0] == positions.shape[0], (
        f"token-count mismatch: q={query.shape[0]} k={key.shape[0]} p={positions.shape[0]}"
    )
    assert query.shape[-1] % head_size == 0 and key.shape[-1] % head_size == 0

    if positions.numel() == 0:
        return

    _rotate_inplace(query, positions, cos_sin_cache, head_size)
    _rotate_inplace(key, positions, cos_sin_cache, head_size)
