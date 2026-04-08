"""
ROCm port validation tests for custom CUDA/HIP kernels.

Covers four genuine blockers that are NOT "just works" and NOT Triton-replaceable,
confirmed against the actual ROCm 6.2 / HIP 6.2.41133 / clang++ 18 headers:

  1. __grid_constant__ undefined in HIP 6.2
     store.cu:28, index.cu:34,65 use __grid_constant__ on kernel parameter structs.
     HIP 6.2 does not define this keyword anywhere (confirmed: grep of all ROCm
     headers returns nothing). Causes an immediate compile error.
     Fix: #define __grid_constant__ /**/ inside an #ifdef __HIP__ guard in utils.cuh,
     or pass -D__grid_constant__= via the JIT build flags on ROCm.

  2. cudaLaunchKernelEx → HIP launch API
     utils.cuh uses cudaLaunchKernelEx / cudaLaunchConfig_t which have no direct
     HIP 6.2 equivalent (hipLaunchKernelEx also absent). Must be replaced with
     hipLaunchKernel or <<<>>> syntax.

  3. --expt-relaxed-constexpr (nvcc-only build flag)
     kernel/utils.py passes this flag unconditionally. hipcc/clang++ will reject it.
     Must be dropped or conditionally excluded for ROCm.

  4. kDLCUDA device-type constraint
     store.cu and index.cu use TensorMatcher::with_device<kDLCUDA>(...) to gate on
     device type. On ROCm, PyTorch tensors are DLPack-exported as kDLROCm (type 10),
     not kDLCUDA (type 2). The device constraint must accept kDLROCm.

Each test is minimal and directly targets one observable failure mode.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_gpu():
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")


# ---------------------------------------------------------------------------
# Test 1  –  store.cu compiles and launches on ROCm
#
# Failure modes (all confirmed against ROCm 6.2 headers):
#   a) __grid_constant__ not defined in HIP 6.2 → compile error on kernel
#      parameter (store.cu:28)
#   b) --expt-relaxed-constexpr rejected by clang++ → compile error
#   c) cudaLaunchKernelEx not in HIP 6.2 (hipLaunchKernelEx also absent)
#      → link error
# ---------------------------------------------------------------------------

def test_store_cache_compiles_and_runs():
    """store_cache JIT-compiles and writes correct values on a ROCm device."""
    _require_gpu()
    from minisgl.kernel import store_cache

    device = torch.device("cuda:0")
    N, H = 1024, 128
    k_cache = torch.zeros(N, H, dtype=torch.float16, device=device)
    v_cache = torch.zeros(N, H, dtype=torch.float16, device=device)
    bs = 32
    indices = torch.randperm(N, device=device)[:bs].to(torch.int32)
    k = torch.randn(bs, H, dtype=torch.float16, device=device)
    v = torch.randn(bs, H, dtype=torch.float16, device=device)

    store_cache(k_cache, v_cache, indices, k, v)
    torch.cuda.synchronize()

    assert torch.all(k_cache[indices] == k), "k_cache mismatch after store_cache"
    assert torch.all(v_cache[indices] == v), "v_cache mismatch after store_cache"


# ---------------------------------------------------------------------------
# Test 2  –  index.cu compiles and launches on ROCm
#
# Failure modes: same as Test 1 plus index.cu:34 and index.cu:65 also use
# __grid_constant__ (three kernel signatures total across both files).
# ---------------------------------------------------------------------------

def test_indexing_compiles_and_runs():
    """indexing JIT-compiles and returns correct embeddings on a ROCm device."""
    _require_gpu()
    from minisgl.kernel import indexing

    device = torch.device("cuda:0")
    V, E = 4096, 256  # vocab × embed
    weights = torch.randn(V, E, dtype=torch.float16, device=device)
    bs = 64
    indices = torch.randint(0, V, (bs,), dtype=torch.int32, device=device)

    result = indexing(weights, indices)
    torch.cuda.synchronize()

    expected = F.embedding(indices.long(), weights)
    assert torch.all(result == expected), "indexing result mismatch"


# ---------------------------------------------------------------------------
# Test 3  –  kDLCUDA device check accepts ROCm tensors
#
# Failure mode: TensorMatcher::with_device<kDLCUDA> rejects a tensor whose
# DLPack device_type is kDLROCm (10) instead of kDLCUDA (2).
# If this check is not broadened to accept kDLROCm, store_cache / indexing
# will raise a "Device mismatch" error at runtime.
#
# The test deliberately creates tensors on every available GPU to cover
# multi-device scenarios.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device_id", range(max(1, torch.cuda.device_count())))
def test_store_cache_device_type_check(device_id: int):
    """store_cache must accept tensors on ROCm GPU regardless of DLPack device_type."""
    _require_gpu()
    from minisgl.kernel import store_cache

    device = torch.device(f"cuda:{device_id}")
    N, H = 256, 64
    k_cache = torch.zeros(N, H, dtype=torch.float16, device=device)
    v_cache = torch.zeros(N, H, dtype=torch.float16, device=device)
    bs = 8
    indices = torch.arange(bs, dtype=torch.int32, device=device)
    k = torch.ones(bs, H, dtype=torch.float16, device=device)
    v = torch.ones(bs, H, dtype=torch.float16, device=device) * 2

    # Must not raise "Tensor must be on CUDA device" or "Device mismatch"
    store_cache(k_cache, v_cache, indices, k, v)
    torch.cuda.synchronize(device)
    assert torch.all(k_cache[:bs] == 1.0)
    assert torch.all(v_cache[:bs] == 2.0)


# ---------------------------------------------------------------------------
# Test 4  –  PDL is disabled on CDNA2
#
# Failure mode A (compile-time): utils.cuh references
#   cudaLaunchAttributeProgrammaticStreamSerialization
# inside a runtime `if (use_pdl)` branch of LaunchKernel::with_attr.
# Even when use_pdl=False at runtime, the symbol must be present for the
# TU to link.  On ROCm/HIP this symbol does not exist.  The ported code
# must guard this branch with #ifdef __CUDA_ARCH__ / #ifndef __HIP__ or
# remove it entirely (PDL has no CDNA2 equivalent).
#
# Failure mode B (run-time): if use_pdl were ever True, the PTX instructions
#   griddepcontrol.wait / griddepcontrol.launch_dependents
# in utils.cuh PDL::wait/launch would be emitted, which are invalid on gfx90a.
#
# This test asserts the Python-side config and that the kernel runs without PDL.
# ---------------------------------------------------------------------------

def test_pdl_disabled_in_default_configs():
    """Default kernel configs must never enable PDL on CDNA2 (gfx90a)."""
    from minisgl.kernel.store import DEFAULT_INDEX_KERNEL_CONFIG as STORE_CFG
    from minisgl.kernel.index import DEFAULT_INDEX_KERNEL_CONFIG as INDEX_CFG

    assert not STORE_CFG.use_pdl, (
        "store kernel DEFAULT config has use_pdl=True — "
        "PDL (griddepcontrol PTX) is not supported on CDNA2/gfx90a"
    )
    assert not INDEX_CFG.use_pdl, (
        "index kernel DEFAULT config has use_pdl=True — "
        "PDL (griddepcontrol PTX) is not supported on CDNA2/gfx90a"
    )


def test_store_cache_runs_without_pdl():
    """Explicitly verify store_cache executes correctly with use_pdl=False config."""
    _require_gpu()
    from minisgl.kernel.store import DEFAULT_INDEX_KERNEL_CONFIG, _jit_store_module

    assert not DEFAULT_INDEX_KERNEL_CONFIG.use_pdl
    device = torch.device("cuda:0")
    N, H = 128, 128
    k_cache = torch.zeros(N, H, dtype=torch.float16, device=device)
    v_cache = torch.zeros(N, H, dtype=torch.float16, device=device)
    indices = torch.arange(4, dtype=torch.int32, device=device)
    k = torch.full((4, H), 3.0, dtype=torch.float16, device=device)
    v = torch.full((4, H), 5.0, dtype=torch.float16, device=device)

    from minisgl.kernel import store_cache
    store_cache(k_cache, v_cache, indices, k, v)
    torch.cuda.synchronize()

    assert torch.all(k_cache[:4] == 3.0)
    assert torch.all(v_cache[:4] == 5.0)


# ---------------------------------------------------------------------------
# Test 5  –  indexing with vocab mask on ROCm
#
# The masked_index_kernel exercises the same HIP launch path but through a
# different code branch (out-of-range tokens → zeroed output).
# ---------------------------------------------------------------------------

def test_indexing_with_vocab_mask():
    """Masked indexing (vocab_range) produces correct zero-masked output on ROCm."""
    _require_gpu()
    from minisgl.kernel import indexing

    device = torch.device("cuda:0")
    V, E = 512, 128
    TP = 2
    weights = torch.randn(V // TP, E, dtype=torch.float16, device=device)
    vocab_start = 0
    vocab_len = V // TP  # local shard covers [0, V//TP)

    bs = 16
    # half in-range, half out-of-range
    in_range = torch.arange(0, bs // 2, dtype=torch.int32, device=device)
    out_of_range = torch.arange(vocab_len, vocab_len + bs // 2, dtype=torch.int32, device=device)
    indices = torch.cat([in_range, out_of_range])

    result = indexing(weights, indices, vocab_range=(vocab_start, vocab_len))
    torch.cuda.synchronize()

    # in-range rows must match weights
    assert torch.all(result[: bs // 2] == weights[in_range])
    # out-of-range rows must be zeroed
    assert torch.all(result[bs // 2 :] == 0), "out-of-range tokens must produce zero rows"
