/** \example example.c
 *  SHAFFT C example using HIP directly.
 *  Demonstrates the C API with explicit GPU memory management.
 */
#include <shafft/shafft.h>

#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int ndim = 3;
  int dims[3] = {64, 64, 32};

  void* plan;
  shafftPlanCreate(&plan);

  // NDA planner
  const int nda = 1;
  shafftPlanNDA(plan, ndim, nda, dims, SHAFFT_C2C, MPI_COMM_WORLD);

  int subsize[ndim], offset[ndim];
  shafftGetLayout(plan, subsize, offset, SHAFFT_TENSOR_LAYOUT_CURRENT);

  size_t elem_count = 0;
  shafftGetAllocSize(plan, &elem_count);

  hipComplex *d_data, *d_work;
  hipMalloc((void**)&d_data, elem_count * sizeof(hipComplex));
  hipMalloc((void**)&d_work, elem_count * sizeof(hipComplex));

  int local_elems = 1;
  for (int i = 0; i < ndim; ++i)
    local_elems *= subsize[i];

  // Initialize with single global impulse at [0,0,0] (only on rank 0)
  hipComplex* h = (hipComplex*)calloc(local_elems, sizeof(hipComplex));
  if (rank == 0) {
    h[0].x = 1.0f;
    h[0].y = 1.0f;
  }
  hipMemcpy(d_data, h, local_elems * sizeof(hipComplex), hipMemcpyHostToDevice);

  shafftSetBuffers(plan, d_data, d_work);
  shafftExecute(plan, SHAFFT_FORWARD);
  shafftNormalize(plan);
  shafftGetBuffers(plan, (void**)&d_data, (void**)&d_work);

  hipMemcpy(h, d_data, local_elems * sizeof(hipComplex), hipMemcpyDeviceToHost);

  // Print result on rank 0
  if (rank == 0 && local_elems >= 4) {
    printf("Result[0..3] = (%g,%g) (%g,%g) (%g,%g) (%g,%g)\n", h[0].x, h[0].y, h[1].x, h[1].y,
           h[2].x, h[2].y, h[3].x, h[3].y);
  }

  free(h);
  hipFree(d_data);
  hipFree(d_work);
  shafftDestroy(&plan);

  MPI_Finalize();
  return 0;
}
