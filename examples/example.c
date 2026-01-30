/** \example example.c
 *  SHAFFT C example using HIP directly.
 *  Demonstrates the C API with explicit GPU memory management.
 */
#include <shafft/shafft.h>
#include <mpi.h>
#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  const int ndim = 3;
  int dims[3] = {64, 64, 32};

  void* plan;
  shafftPlanCreate(&plan);

  // NDA planner
  const int nda = 1;
  shafftPlanNDA(plan, ndim, nda, dims, SHAFFT_C2C, MPI_COMM_WORLD);

  // --- Alternative: Cartesian planner ---
  /*
  int subsize[ndim], offset[ndim], COMM_DIMS[ndim] = {0};
  int COMM_SIZE;
  size_t mem_limit = 0;
  shafftConfigurationCart(ndim, dims, subsize, offset, COMM_DIMS, &COMM_SIZE,
                          SHAFFT_C2C, mem_limit, MPI_COMM_WORLD);
  shafftPlanCart(plan, ndim, COMM_DIMS, dims, SHAFFT_C2C, MPI_COMM_WORLD);
  */

  int subsize[ndim], offset[ndim];
  shafftGetLayout(plan, subsize, offset, SHAFFT_TENSOR_LAYOUT_CURRENT);

  size_t elem_count = 0;
  shafftGetAllocSize(plan, elem_count);

  hipComplex *d_data, *d_work;
  hipMalloc((void**)&d_data, elem_count * sizeof(hipComplex));
  hipMalloc((void**)&d_work, elem_count * sizeof(hipComplex));

  int local_elems = 1;
  for (int i = 0; i < ndim; ++i) local_elems *= subsize[i];

  hipComplex* h = (hipComplex*)calloc(local_elems, sizeof(hipComplex));
  h[0].x = 1.0f; h[0].y = 1.0f;
  hipMemcpy(d_data, h, local_elems * sizeof(hipComplex), hipMemcpyHostToDevice);

  shafftSetBuffers(plan, d_data, d_work);
  shafftExecute(plan, SHAFFT_FORWARD);
  shafftNormalize(plan);
  shafftGetBuffers(plan, (void**)&d_data, (void**)&d_work);

  hipMemcpy(h, d_data, local_elems * sizeof(hipComplex), hipMemcpyDeviceToHost);

  free(h);
  hipFree(d_data);
  hipFree(d_work);
  shafftDestroy(plan);

  MPI_Finalize();
  return 0;
}
