/**
 * @file buffer_utils.hpp
 * @brief Portable buffer allocation, deallocation, and copy utilities.
 *
 * Provides backend-agnostic memory operations that use HIP on GPU builds
 * and standard library functions on CPU builds.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef SHAFFT_DETAIL_BUFFER_UTILS_HPP
#define SHAFFT_DETAIL_BUFFER_UTILS_HPP

#include <shafft/shafft_enums.hpp>

#include <cstdlib>
#include <cstring>

#if SHAFFT_BACKEND_HIPFFT
#include <hip/hip_runtime.h>
#endif

namespace shafft::detail {

/**
 * @brief Allocate a buffer of count elements of type T.
 *
 * On GPU builds, allocates device memory via hipMalloc.
 * On CPU builds, allocates host memory via std::malloc.
 *
 * @tparam T Element type
 * @param count Number of elements to allocate
 * @param buf   [out] Pointer to receive the allocated buffer
 * @return SHAFFT_SUCCESS on success, error code on failure
 */
template <typename T>
int allocBufferT(size_t count, T** buf) noexcept {
  if (!buf)
    return static_cast<int>(Status::ERR_NULLPTR);
  if (count == 0) {
    *buf = nullptr;
    return static_cast<int>(Status::SUCCESS);
  }
#if SHAFFT_BACKEND_HIPFFT
  hipError_t err = hipMalloc(reinterpret_cast<void**>(buf), count * sizeof(T));
  if (err != hipSuccess) {
    *buf = nullptr;
    return static_cast<int>(Status::ERR_ALLOC);
  }
#else
  *buf = static_cast<T*>(std::malloc(count * sizeof(T)));
  if (!*buf)
    return static_cast<int>(Status::ERR_ALLOC);
#endif
  return static_cast<int>(Status::SUCCESS);
}

/**
 * @brief Free a buffer previously allocated with allocBufferT.
 *
 * @tparam T Element type
 * @param buf Pointer to the buffer to free (nullptr is safe)
 * @return SHAFFT_SUCCESS on success, error code on failure
 */
template <typename T>
int freeBufferT(T* buf) noexcept {
  if (!buf)
    return static_cast<int>(Status::SUCCESS);
#if SHAFFT_BACKEND_HIPFFT
  hipError_t err = hipFree(buf);
  if (err != hipSuccess)
    return static_cast<int>(Status::ERR_BACKEND);
#else
  std::free(buf);
#endif
  return static_cast<int>(Status::SUCCESS);
}

/**
 * @brief Copy data from host to buffer (device on GPU, host on CPU).
 *
 * @tparam T Element type
 * @param dst   Destination buffer
 * @param src   Source buffer (host memory)
 * @param count Number of elements to copy
 * @return SHAFFT_SUCCESS on success, error code on failure
 */
template <typename T>
int copyToBufferT(T* dst, const T* src, size_t count) noexcept {
  if (!dst || !src)
    return static_cast<int>(Status::ERR_NULLPTR);
  if (count == 0)
    return static_cast<int>(Status::SUCCESS);
#if SHAFFT_BACKEND_HIPFFT
  hipError_t err = hipMemcpy(dst, src, count * sizeof(T), hipMemcpyHostToDevice);
  if (err != hipSuccess)
    return static_cast<int>(Status::ERR_BACKEND);
#else
  std::memcpy(dst, src, count * sizeof(T));
#endif
  return static_cast<int>(Status::SUCCESS);
}

/**
 * @brief Copy data from buffer to host (device on GPU, host on CPU).
 *
 * @tparam T Element type
 * @param dst   Destination buffer (host memory)
 * @param src   Source buffer
 * @param count Number of elements to copy
 * @return SHAFFT_SUCCESS on success, error code on failure
 */
template <typename T>
int copyFromBufferT(T* dst, const T* src, size_t count) noexcept {
  if (!dst || !src)
    return static_cast<int>(Status::ERR_NULLPTR);
  if (count == 0)
    return static_cast<int>(Status::SUCCESS);
#if SHAFFT_BACKEND_HIPFFT
  hipError_t err = hipMemcpy(dst, src, count * sizeof(T), hipMemcpyDeviceToHost);
  if (err != hipSuccess)
    return static_cast<int>(Status::ERR_BACKEND);
#else
  std::memcpy(dst, src, count * sizeof(T));
#endif
  return static_cast<int>(Status::SUCCESS);
}

} // namespace shafft::detail

#endif // SHAFFT_DETAIL_BUFFER_UTILS_HPP
