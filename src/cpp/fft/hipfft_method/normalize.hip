#include <vector>
#include <numeric>
#include <cassert>
#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include "hipcheck.h"

template <typename T, typename U>
__global__ void normalize(U norm_factor, size_t tensorSize, T *odata)
{
  int grid_size = blockDim.x * gridDim.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = idx; i < tensorSize; i += grid_size)
  {
    odata[i].x *= norm_factor;
    odata[i].y *= norm_factor;
  }
}

int normalizeComplexFloat(float norm_factor, size_t tensor_size, void *data){
  hipLaunchKernelGGL((normalize<hipComplex, float>),
                     dim3(1200), dim3(256), 0, 0,
                     norm_factor, tensor_size, (hipComplex *)data);
  return hipCheck(hipDeviceSynchronize());
}

int normalizeComplexDouble(double norm_factor, size_t tensor_size, void *data){
  hipLaunchKernelGGL((normalize<hipDoubleComplex, double>),
                     dim3(1200), dim3(256), 0, 0,
                     norm_factor, tensor_size, (hipDoubleComplex *)data);
  return hipCheck(hipDeviceSynchronize());
}
