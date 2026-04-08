# ROCm Porting Guide

**Baseline**: ROCm 6.2 / HIP 6.2.41133 / clang++ 18 / RCCL 2.20.5 / CDNA2 (gfx90a)

---

## Overview

Three categories of items are **excluded** from this guide because they require no
special porting effort:

- **Just works**: `torch.cuda.*` (CUDA Graph, Stream, synchronize, nvtx, mem_get_info)
  all function transparently under ROCm PyTorch. No code changes needed.
- **Triton-replaceable**: `flashinfer` fused ops (rmsnorm, silu_and_mul, gelu_and_mul,
  rotary, sampling) and `sgl_kernel` attention/MoE kernels have no ROCm port but can
  be replaced with Triton kernels or `torch`-native equivalents.

What remains are **four hard blockers** — items that will cause a compile or link error
on ROCm 6.2 with zero workaround other than a targeted source change.

---

## Step 1 — `__grid_constant__` undefined in HIP 6.2

### Background

`__grid_constant__` is a CUDA 11.7+ keyword that marks a kernel parameter struct as
read-only in constant memory for the duration of the grid. clang++ / HIP 6.2 does not
define it anywhere (confirmed: `grep -r __grid_constant__ /opt/rocm/include/` returns
nothing).

### Affected files

| File | Line(s) |
|------|---------|
| `python/minisgl/kernel/csrc/jit/store.cu` | 28 |
| `python/minisgl/kernel/csrc/jit/index.cu` | 34, 65 |

### Required change

Add to `python/minisgl/kernel/csrc/include/minisgl/utils.cuh` (near the top, after
system includes):

```cpp
#ifdef __HIP__
#  define __grid_constant__   // no constant-memory parameter cache on CDNA2
#endif
```

No correctness impact: on CDNA2 the struct is passed by value instead of via
constant memory. Performance is equivalent because CDNA2 scalar registers serve
the same purpose.

### Validation test

```
tests/rocm/test_custom_kernels.py::test_store_cache_compiles_and_runs
tests/rocm/test_custom_kernels.py::test_indexing_compiles_and_runs
```

---

## Step 2 — `cudaLaunchKernelEx` / `cudaLaunchConfig_t` → HIP launch API

### Background

`LaunchKernel::operator()` in `utils.cuh` uses the CUDA 12 extended launch API:

```cpp
CUDA_CHECK(::cudaLaunchKernelEx(&m_config, kernel, args...));
```

`cudaLaunchKernelEx` and `cudaLaunchConfig_t` have no equivalent in HIP 6.2
(`hipLaunchKernelEx` is also absent from HIP 6.2 headers). The existing
`hipLaunchKernel` / triple-chevron `<<<>>>` syntax covers the required
functionality (grid, block, smem, stream).

### Affected files

| File | Symbol |
|------|--------|
| `python/minisgl/kernel/csrc/include/minisgl/utils.cuh` | `LaunchKernel::operator()`, `LaunchKernel::s_make_config`, `cudaLaunchConfig_t m_config`, `cudaLaunchAttribute m_attr_cache` |

### Required change

Replace the `LaunchKernel` internals with a HIP-compatible implementation.
The grid/block/smem/stream parameters map 1-to-1; only the attribute
machinery (used exclusively for PDL, see Step 3) needs removal.

```cpp
// utils.cuh — replace LaunchKernel with:
struct LaunchKernel {
public:
  explicit LaunchKernel(dim3 grid_dim, dim3 block_dim, DLDevice device,
                        std::size_t dynamic_shared_mem_bytes = 0) noexcept
      : m_grid(grid_dim), m_block(block_dim),
        m_smem(dynamic_shared_mem_bytes),
        m_stream(resolve_device(device)) {}

  static auto resolve_device(DLDevice device) -> hipStream_t {
    return static_cast<hipStream_t>(
        ::TVMFFIEnvGetStream(device.device_type, device.device_id));
  }

  template <typename T, typename... Args>
  auto operator()(T kernel, Args &&...args) const -> void {
    kernel<<<m_grid, m_block, m_smem, m_stream>>>(
        std::forward<Args>(args)...);
    HIP_CHECK(hipGetLastError());
  }

  // PDL has no CDNA2 equivalent — with_attr is a no-op on ROCm
  auto with_attr(bool /*use_pdl*/) -> LaunchKernel & { return *this; }

private:
  dim3        m_grid, m_block;
  std::size_t m_smem;
  hipStream_t m_stream;
};
```

Also replace `CUDA_CHECK` with `HIP_CHECK` using `hipGetLastError` / `hipSuccess`.

### Validation test

```
tests/rocm/test_custom_kernels.py::test_store_cache_compiles_and_runs
tests/rocm/test_custom_kernels.py::test_indexing_compiles_and_runs
tests/rocm/test_custom_kernels.py::test_store_cache_runs_without_pdl
tests/rocm/test_custom_kernels.py::test_indexing_with_vocab_mask
```

---

## Step 3 — PDL host-side symbols (`cudaLaunchAttributeProgrammaticStreamSerialization`)

### Background

Even with `use_pdl=False` at runtime, `LaunchKernel::with_attr` in `utils.cuh`
contains a runtime `if (use_pdl)` branch (not `if constexpr`) that references
`cudaLaunchAttributeProgrammaticStreamSerialization`. The compiler resolves all
symbols in a compiled TU regardless of runtime branch taken.

HIP 6.2's `hipLaunchAttributeID` enum has only three entries
(AccessPolicyWindow=1, Cooperative=2, Priority=8) — no PDL equivalent exists.

The device-side PTX (`griddepcontrol.wait/launch_dependents`) inside
`PDL::wait<kUsePDL>()` / `PDL::launch<kUsePDL>()` is gated by
`if constexpr (kUsePDL)`. Since `kUsePDL=false` always on CDNA2, clang++
discards those branches entirely before ASM parsing — this is **not** a problem.

### Affected files

| File | Location |
|------|----------|
| `python/minisgl/kernel/csrc/include/minisgl/utils.cuh` | `LaunchKernel::with_attr`, `PDL::wait`, `PDL::launch` |
| `python/minisgl/kernel/utils.py` | `KernelConfig.use_pdl` field |
| `python/minisgl/kernel/store.py` | `DEFAULT_INDEX_KERNEL_CONFIG` |
| `python/minisgl/kernel/index.py` | `DEFAULT_INDEX_KERNEL_CONFIG` |

### Required change

Step 2's `LaunchKernel` replacement already makes `with_attr` a no-op, which
resolves the compile error. Separately, assert at Python import time that
`use_pdl` is never `True` on a ROCm build:

```python
# kernel/utils.py — add after KernelConfig definition:
import torch
if torch.version.hip is not None:
    # PDL (griddepcontrol PTX) has no CDNA equivalent
    _original_KernelConfig = KernelConfig
    class KernelConfig(_original_KernelConfig):
        def __new__(cls, num_threads, max_occupancy, use_pdl):
            assert not use_pdl, \
                "use_pdl=True is not supported on ROCm (no PDL on CDNA2/gfx90a)"
            return super().__new__(cls, num_threads, max_occupancy, False)
```

### Validation test

```
tests/rocm/test_custom_kernels.py::test_pdl_disabled_in_default_configs
tests/rocm/test_custom_kernels.py::test_store_cache_runs_without_pdl
```

---

## Step 4 — `--expt-relaxed-constexpr` (nvcc-only build flag)

### Background

`python/minisgl/kernel/utils.py` passes `--expt-relaxed-constexpr` in
`DEFAULT_CUDA_CFLAGS` unconditionally. This flag allows `constexpr` functions in
device code under nvcc. clang++ / hipcc rejects unknown flags with a hard error.
(clang++ 18 allows `constexpr` in device code natively — no flag needed.)

### Affected files

| File | Line |
|------|------|
| `python/minisgl/kernel/utils.py` | `DEFAULT_CUDA_CFLAGS = ["-std=c++20", "-O3", "--expt-relaxed-constexpr"]` |

### Required change

```python
# kernel/utils.py
import torch

_BASE_CUDA_CFLAGS = ["-std=c++20", "-O3"]
DEFAULT_CUDA_CFLAGS = (
    _BASE_CUDA_CFLAGS
    if torch.version.hip is not None          # clang++ supports constexpr natively
    else _BASE_CUDA_CFLAGS + ["--expt-relaxed-constexpr"]
)
```

### Validation test

```
tests/rocm/test_custom_kernels.py::test_store_cache_compiles_and_runs
tests/rocm/test_custom_kernels.py::test_indexing_compiles_and_runs
```

*(Any JIT kernel compile test exercises this path — a compile error would surface
immediately as an exception from `load_jit`.)*

---

## Step 5 — `kDLCUDA` device-type constraint rejects ROCm tensors

### Background

`store.cu` and `index.cu` use `TensorMatcher::with_device<kDLCUDA>(device_)` to
constrain input tensors to CUDA devices. Under ROCm, PyTorch exports tensors via
DLPack with `device_type = kDLROCm` (value 10), not `kDLCUDA` (value 2). The
`TensorMatcher` will throw "Device mismatch" at runtime.

`pynccl.cu` has the same issue: `RuntimeCheck(t.device().device_type == kDLCUDA,
"Tensor must be on CUDA device")`.

### Affected files

| File | Location |
|------|----------|
| `python/minisgl/kernel/csrc/jit/store.cu` | `TensorMatcher({-1,D}).with_device<kDLCUDA>` (×2) |
| `python/minisgl/kernel/csrc/jit/index.cu` | `TensorMatcher({-1,D}).with_device<kDLCUDA>` (×3) |
| `python/minisgl/kernel/csrc/src/pynccl.cu` | `RuntimeCheck(... == kDLCUDA ...)` (×4) |

### Required change

Add a `kDLGPU` alias in `utils.cuh` or `tensor.h` that covers both device types:

```cpp
// utils.cuh or a new compat header
#ifdef __HIP__
inline constexpr DLDeviceType kDLGPU = kDLROCm;
#else
inline constexpr DLDeviceType kDLGPU = kDLCUDA;
#endif
```

Then replace all `kDLCUDA` device constraints and checks with `kDLGPU`:

```cpp
// store.cu, index.cu: change
TensorMatcher({-1, D}).with_device<kDLCUDA>(device_)
// to
TensorMatcher({-1, D}).with_device<kDLGPU>(device_)

// pynccl.cu: change
RuntimeCheck(t.device().device_type == kDLCUDA, "Tensor must be on CUDA device");
// to
RuntimeCheck(t.device().device_type == kDLGPU, "Tensor must be on GPU device");
```

Also update `LaunchKernel::resolve_device` which calls
`TVMFFIEnvGetStream(device.device_type, device.device_id)` — ensure TVM-FFI is
configured to resolve `kDLROCm` to a HIP stream (this may be handled by TVM-FFI
itself; verify at integration time).

### Validation test

```
tests/rocm/test_custom_kernels.py::test_store_cache_device_type_check[0]
tests/rocm/test_custom_kernels.py::test_store_cache_device_type_check[N]  (all GPUs)
```

---

## Step 6 — RCCL missing NCCL 2.27 window APIs

### Background

`pynccl.cu`'s `NCCLWrapper` constructor calls NCCL 2.27 symmetric-memory APIs:

```cpp
ncclMemAlloc(&buf, max_bytes);                              // RCCL 2.20 ✓
ncclCommWindowRegister(comm, buf, max_bytes, &win, ...);    // RCCL 2.20 ✗
ncclCommWindowDeregister(comm, win);                        // RCCL 2.20 ✗
```

RCCL bundled with ROCm 6.2 is version **2.20.5**. `ncclMemAlloc`/`ncclMemFree`
are present; `ncclCommWindowRegister`, `ncclCommWindowDeregister`, `ncclWindow_t`,
and `NCCL_WIN_COLL_SYMMETRIC` are absent (confirmed: no matches in
`/opt/rocm/include/rccl/rccl.h`).

The symmetric buffer is an optimisation for small AllReduces that fit in the
pre-registered window. On ROCm, all AllReduces must take the fallback path
(direct pointer, no window).

Also, the bundled `nccl227.h` includes `<cuda_runtime.h>` at the top.
`/opt/rocm` does not ship a standalone `cuda_runtime.h`; pynccl.cu must include
`<rccl/rccl.h>` and `<hip/hip_runtime.h>` directly on ROCm.

### Affected files

| File | Change needed |
|------|--------------|
| `python/minisgl/kernel/csrc/src/pynccl.cu` | Guard window APIs; switch header |
| `python/minisgl/kernel/csrc/include/minisgl/nccl227.h` | Used only on CUDA; replace on ROCm |

### Required change

```cpp
// pynccl.cu — replace top includes:
#ifdef __HIP__
#  include <rccl/rccl.h>
#  include <hip/hip_runtime.h>
#else
#  include <minisgl/nccl227.h>
#endif

// NCCLWrapper constructor — guard window registration:
NCCLWrapper(int rank, int world_size, size_t max_bytes, NCCLIDList uid)
    : m_rank(rank), m_world_size(world_size), m_max_bytes(max_bytes) {
  ncclUniqueId id = get_uid(uid);
  ncclComm_t comm;
  NCCL_CHECK(::ncclCommInitRank(&comm, m_world_size, id, m_rank));
  m_comm = {comm, template_fn<::ncclCommDestroy>};

  void *buf = nullptr;
#if NCCL_VERSION_CODE >= 22600   // window APIs introduced in NCCL 2.26
  if (max_bytes > 0) {
    NCCL_CHECK(::ncclMemAlloc(&buf, max_bytes));
    m_sym_mem = {buf, template_fn<::ncclMemFree>};
    ncclWindow_t win;
    NCCL_CHECK(::ncclCommWindowRegister(comm, buf, max_bytes, &win,
                                        NCCL_WIN_COLL_SYMMETRIC));
    m_win = {win, [comm = m_comm](ncclWindow_t w) {
               return NCCL_CHECK(::ncclCommWindowDeregister(comm.get(), w));
             }};
  }
#else
  // RCCL / pre-2.26 NCCL: allocate via hipMalloc, no window registration
  if (max_bytes > 0) {
    HIP_CHECK(hipMalloc(&buf, max_bytes));
    m_sym_mem = {buf, [](void *p) { hipFree(p); }};
  }
#endif
}
```

Remove `m_win` member on the `#else` path (or make it `std::monostate`).

### Validation test

```
tests/rocm/test_rccl.py::test_rccl_allreduce_allgather_2gpu[2]
tests/rocm/test_rccl.py::test_rccl_allreduce_allgather_4gpu[4]
```

---

## Summary table

| Step | File(s) changed | Error without fix | Validation test(s) |
|------|----------------|-------------------|--------------------|
| 1 · `__grid_constant__` | `utils.cuh` | Compile error | `test_store_cache_compiles_and_runs`, `test_indexing_compiles_and_runs` |
| 2 · `cudaLaunchKernelEx` | `utils.cuh` | Link error | `test_store_cache_compiles_and_runs`, `test_indexing_compiles_and_runs`, `test_store_cache_runs_without_pdl`, `test_indexing_with_vocab_mask` |
| 3 · PDL host symbols | `utils.cuh`, `kernel/utils.py` | Compile error | `test_pdl_disabled_in_default_configs`, `test_store_cache_runs_without_pdl` |
| 4 · `--expt-relaxed-constexpr` | `kernel/utils.py` | Compile error (clang++ rejects unknown flag) | `test_store_cache_compiles_and_runs`, `test_indexing_compiles_and_runs` |
| 5 · `kDLCUDA` device check | `store.cu`, `index.cu`, `pynccl.cu` | Runtime "Device mismatch" error | `test_store_cache_device_type_check[*]` |
| 6 · RCCL window APIs | `pynccl.cu`, `nccl227.h` | Link error (`ncclCommWindowRegister` undefined) | `test_rccl_allreduce_allgather_2gpu`, `test_rccl_allreduce_allgather_4gpu` |

---

## What was explicitly excluded

### Just works — no code changes needed

| Item | Why no change needed |
|------|---------------------|
| `torch.cuda.CUDAGraph`, `torch.cuda.graph`, `torch.cuda.Stream` | ROCm PyTorch maps these to HIP equivalents transparently |
| `torch.cuda.synchronize`, `torch.cuda.mem_get_info`, `torch.cuda.empty_cache` | Same — transparent HIP mapping |
| `torch.cuda.nvtx` | Maps to ROCTx in ROCm PyTorch builds |
| `ncclMemAlloc` / `ncclMemFree` | Present in RCCL 2.20.5 (confirmed) |
| PDL device-side PTX (`griddepcontrol.*`) | Inside `if constexpr (kUsePDL=false)` — clang++ discards before ASM parsing |
| C++20 (`std::source_location`, `std::bit_cast`, concepts) | clang++ 18 in ROCm 6.2 fully supports |
| NCCL → RCCL library swap | Drop-in: same API surface, just link `-lrccl` |

### Triton/alternative-replaceable — out of scope for this porting step

| Item | Replacement |
|------|-------------|
| `flashinfer.rmsnorm`, `fused_add_rmsnorm` | Triton RMSNorm kernel |
| `flashinfer.silu_and_mul`, `gelu_and_mul` | Triton activation kernel |
| `flashinfer.apply_rope_with_cos_sin_cache_inplace` | Triton RoPE kernel |
| `flashinfer` attention backends (fi.py, trtllm.py) | `flash-attention-rocm` or Triton attention |
| `sgl_kernel.flash_attn` (fa.py) | `flash-attention-rocm` |
| `sgl_kernel.topk_softmax`, `moe_align_block_size` | Triton MoE kernels |
| `flashinfer.sampling.softmax` | Triton / pure PyTorch |
| SM-number arch detection (`is_sm90_supported`, `is_sm100_supported`) | Replace with CDNA arch detection via `torch.cuda.get_device_properties` |
| Dockerfile base image | `rocm/dev-ubuntu-*` |
