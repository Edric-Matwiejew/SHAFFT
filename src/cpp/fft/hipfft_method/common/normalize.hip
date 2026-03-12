#include "hipcheck.hpp"

#include "../../detail/error_macros.hpp"

#include <cassert>
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>

template <typename T, typename U>
__global__ void normalize(U normFactor, size_t tensorSize, T* odata) {
  size_t gridSize = static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);
  size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) +
               static_cast<size_t>(threadIdx.x);
  for (size_t i = idx; i < tensorSize; i += gridSize) {
    odata[i].x *= normFactor;
    odata[i].y *= normFactor;
  }
}

[[nodiscard]] static inline int computeGridDim(size_t tensorSize, dim3& grid, int blockSize = 256) {
  int device;
  SHAFFT_HIP_CHECK(hipGetDevice(&device));
  hipDeviceProp_t props;
  SHAFFT_HIP_CHECK(hipGetDeviceProperties(&props, device));

  size_t targetThreads = tensorSize;
  size_t numBlocks = (targetThreads + blockSize - 1) / blockSize;

  size_t maxBlocks = static_cast<size_t>(props.maxGridSize[0]);
  if (numBlocks < 1)
    numBlocks = 1;
  if (numBlocks > maxBlocks)
    numBlocks = maxBlocks;
  if (numBlocks > 65535)
    numBlocks = 65535; // Conservative limit

  grid = dim3(static_cast<unsigned int>(numBlocks));
  return SHAFFT_STATUS(SHAFFT_SUCCESS);
}

int normalizeComplexFloat(float normFactor, size_t tensorSize, void* data, hipStream_t stream) {
  constexpr int blockSize = 256;
  dim3 grid;
  int rc = computeGridDim(tensorSize, grid, blockSize);
  if (rc != SHAFFT_STATUS(SHAFFT_SUCCESS))
    return rc;

  hipLaunchKernelGGL((normalize<hipComplex, float>),
                     grid,
                     dim3(blockSize),
                     0,
                     stream,
                     normFactor,
                     tensorSize,
                     (hipComplex*)data);
  // Return immediately - no sync. Caller handles synchronization if needed.
  return hipCheck(hipGetLastError());
}

int normalizeComplexDouble(double normFactor, size_t tensorSize, void* data, hipStream_t stream) {
  constexpr int blockSize = 256;
  dim3 grid;
  int rc = computeGridDim(tensorSize, grid, blockSize);
  if (rc != SHAFFT_STATUS(SHAFFT_SUCCESS))
    return rc;

  hipLaunchKernelGGL((normalize<hipDoubleComplex, double>),
                     grid,
                     dim3(blockSize),
                     0,
                     stream,
                     normFactor,
                     tensorSize,
                     (hipDoubleComplex*)data);
  // Return immediately - no sync. Caller handles synchronization if needed.
  return hipCheck(hipGetLastError());
}
