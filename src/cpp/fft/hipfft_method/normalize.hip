#include <vector>
#include <numeric>
#include <cassert>
#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include "hipcheck.h"

template <typename T, typename U>
__global__ void normalize(U norm_factor, size_t tensorSize, T *odata)
{
  size_t grid_size = static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);
  size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
  for (size_t i = idx; i < tensorSize; i += grid_size)
  {
    odata[i].x *= norm_factor;
    odata[i].y *= norm_factor;
  }
}

// Compute optimal grid dimensions based on tensor size
// Target: each thread processes ~4-8 elements for good occupancy vs overhead balance
static inline dim3 compute_grid_dim(size_t tensor_size, int block_size = 256) {
  // Get device properties for max grid size
  int device;
  hipGetDevice(&device);
  hipDeviceProp_t props;
  hipGetDeviceProperties(&props, device);
  
  // Aim for each thread to process ~4 elements (good balance)
  size_t target_threads = (tensor_size + 3) / 4;
  size_t num_blocks = (target_threads + block_size - 1) / block_size;
  
  // Clamp to device limits and reasonable range [1, maxGridSize or 65535]
  size_t max_blocks = static_cast<size_t>(props.maxGridSize[0]);
  if (num_blocks < 1) num_blocks = 1;
  if (num_blocks > max_blocks) num_blocks = max_blocks;
  if (num_blocks > 65535) num_blocks = 65535;  // Conservative limit
  
  return dim3(static_cast<unsigned int>(num_blocks));
}

int normalizeComplexFloat(float norm_factor, size_t tensor_size, void *data, hipStream_t stream){
  constexpr int block_size = 256;
  dim3 grid = compute_grid_dim(tensor_size, block_size);
  
  hipLaunchKernelGGL((normalize<hipComplex, float>),
                     grid, dim3(block_size), 0, stream,
                     norm_factor, tensor_size, (hipComplex *)data);
  // Return immediately - no sync. Caller handles synchronization if needed.
  return hipCheck(hipGetLastError());
}

int normalizeComplexDouble(double norm_factor, size_t tensor_size, void *data, hipStream_t stream){
  constexpr int block_size = 256;
  dim3 grid = compute_grid_dim(tensor_size, block_size);
  
  hipLaunchKernelGGL((normalize<hipDoubleComplex, double>),
                     grid, dim3(block_size), 0, stream,
                     norm_factor, tensor_size, (hipDoubleComplex *)data);
  // Return immediately - no sync. Caller handles synchronization if needed.
  return hipCheck(hipGetLastError());
}
