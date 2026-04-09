#pragma once

#include <minisgl/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <concepts>
#include <cstddef>
#include <source_location>
#include <type_traits>

namespace device {

inline constexpr auto kWarpThreads = 32u;

template <std::integral T, std::integral U>
__always_inline __device__ constexpr auto div_ceil(T a, U b) {
  return (a + b - 1) / b;
}

namespace pointer {

// we only allow void * pointer arithmetic for safety

template <typename T, std::integral... U>
__always_inline __device__ auto offset(T *ptr, U... offset) -> void * {
  static_assert(std::is_same_v<T, void>,
                "Pointer arithmetic is only allowed for void* pointers");
  return static_cast<char *>(ptr) + (... + offset);
}

template <typename T, std::integral... U>
__always_inline __device__ auto offset(const T *ptr, U... offset) -> const
    void * {
  static_assert(std::is_same_v<T, void>,
                "Pointer arithmetic is only allowed for void* pointers");
  return static_cast<const char *>(ptr) + (... + offset);
}

} // namespace pointer

namespace PDL {

template <bool kUsePDL> __always_inline __device__ void wait() {
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
}

template <bool kUsePDL> __always_inline __device__ void launch() {
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.launch_dependents;" :::);
  }
}

} // namespace PDL

} // namespace device

// ---------------------------------------------------------------------------
// HIP / ROCm compatibility
// ---------------------------------------------------------------------------
#if defined(__HIP__) || defined(MINISGL_ROCM)
#include <hip/hip_runtime.h>

// HIP on AMD does not ship CUDA-compat headers (those live only in
// hip/nvidia_detail/).  Provide the minimal aliases that utils.cuh needs.
using cudaError_t  = hipError_t;
using cudaStream_t = hipStream_t;
inline constexpr hipError_t cudaSuccess = hipSuccess;
inline const char* cudaGetErrorString(hipError_t e) { return hipGetErrorString(e); }
inline hipError_t  cudaGetLastError()               { return hipGetLastError(); }
inline hipError_t  cudaFuncSetAttribute(const void* f, hipFuncAttribute a, int v) {
  return hipFuncSetAttribute(f, a, v);
}
inline constexpr hipFuncAttribute cudaFuncAttributeMaxDynamicSharedMemorySize =
    hipFuncAttributeMaxDynamicSharedMemorySize;

// Memcpy helpers
using cudaMemcpyKind = hipMemcpyKind;
inline constexpr hipMemcpyKind cudaMemcpyDeviceToDevice = hipMemcpyDeviceToDevice;
inline constexpr hipMemcpyKind cudaMemcpyHostToDevice   = hipMemcpyHostToDevice;
inline constexpr hipMemcpyKind cudaMemcpyDeviceToHost   = hipMemcpyDeviceToHost;
inline hipError_t cudaMemcpyAsync(void* dst, const void* src, std::size_t count,
                                  hipMemcpyKind kind, hipStream_t stream) {
  return hipMemcpyAsync(dst, src, count, kind, stream);
}

// __grid_constant__ is a CUDA-specific kernel-parameter qualifier; no-op on HIP.
#ifndef __grid_constant__
#define __grid_constant__
#endif
#endif // __HIP__ or MINISGL_ROCM

namespace host {

inline auto
CUDA_CHECK(::cudaError_t error,
           std::source_location location = std::source_location::current())
    -> void {
  if (error != ::cudaSuccess) {
    [[unlikely]];
    ::host::panic(location, "CUDA error: ", ::cudaGetErrorString(error));
  }
}

inline auto
CUDA_CHECK(std::source_location location = std::source_location::current())
    -> void {
  return CUDA_CHECK(::cudaGetLastError(), location);
}

template <auto F> inline void set_smem_once(std::size_t smem_size) {
  static const auto last_smem_size = [&] {
    CUDA_CHECK(::cudaFuncSetAttribute(
        F, ::cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    return smem_size;
  }();
  RuntimeCheck(
      smem_size <= last_smem_size,
      "Dynamic shared memory size exceeds the previously set maximum size: ",
      last_smem_size, " bytes");
}

struct LaunchKernel {
public:
  explicit LaunchKernel(dim3 grid_dim, dim3 block_dim, DLDevice device,
                        std::size_t dynamic_shared_mem_bytes = 0) noexcept
      : m_grid(grid_dim), m_block(block_dim), m_smem(dynamic_shared_mem_bytes),
        m_stream(resolve_device(device)) {}

  explicit LaunchKernel(dim3 grid_dim, dim3 block_dim, cudaStream_t stream,
                        std::size_t dynamic_shared_mem_bytes = 0) noexcept
      : m_grid(grid_dim), m_block(block_dim), m_smem(dynamic_shared_mem_bytes),
        m_stream(stream) {}

  static auto resolve_device(DLDevice device) -> cudaStream_t {
    return static_cast<cudaStream_t>(
        ::TVMFFIEnvGetStream(device.device_type, device.device_id));
  }

  LaunchKernel(const LaunchKernel &) = delete;
  LaunchKernel &operator=(const LaunchKernel &) = delete;

  template <typename T, typename... Args>
  auto operator()(T &&kernel, Args &&...args) const -> void {
#ifdef __HIP__
    // hipcc does not have hipLaunchKernelEx / cudaLaunchKernelEx.
    // Use the standard <<<>>> triple-chevron syntax instead.
    kernel<<<m_grid, m_block, m_smem, m_stream>>>(std::forward<Args>(args)...);
    CUDA_CHECK();
#else
    CUDA_CHECK(
        ::cudaLaunchKernelEx(&m_config, kernel, std::forward<Args>(args)...));
#endif
  }

  auto with_attr(bool use_pdl) -> LaunchKernel & {
#ifndef __HIP__
    // cudaLaunchAttributeProgrammaticStreamSerialization is CUDA/sm90+ only.
    if (use_pdl) {
      m_attr_cache.id = ::cudaLaunchAttributeProgrammaticStreamSerialization;
      m_attr_cache.val.programmaticStreamSerializationAllowed = 1;
      m_config.attrs = &m_attr_cache;
      m_config.numAttrs = 1;
    } else {
      m_config.numAttrs = 0;
    }
#endif
    return *this;
  }

private:
  dim3 m_grid;
  dim3 m_block;
  std::size_t m_smem;
  cudaStream_t m_stream;
#ifndef __HIP__
  // cudaLaunchConfig_t / cudaLaunchAttribute are CUDA-only.
  cudaLaunchConfig_t m_config = s_make_config(m_grid, m_block, m_stream, m_smem);
  cudaLaunchAttribute m_attr_cache;

  static auto s_make_config(dim3 grid_dim, dim3 block_dim, cudaStream_t stream,
                            std::size_t smem) -> cudaLaunchConfig_t {
    auto config = ::cudaLaunchConfig_t{};
    config.gridDim = grid_dim;
    config.blockDim = block_dim;
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    config.numAttrs = 0;
    return config;
  }
#endif
};

} // namespace host
