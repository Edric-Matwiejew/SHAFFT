/// \example example_portable.c
/// Backend-portable SHAFFT C example.
#include <shafft/shafft.h>

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int dims[3] = {64, 64, 32};
  int commDims[3] = {0, 0, 0};
  size_t subsize[3], offset[3];
  int nda = 0, commSize, rc;

  rc = shafftConfigurationND(3,
                             dims,
                             SHAFFT_C2C,
                             commDims,
                             &nda,
                             subsize,
                             offset,
                             &commSize,
                             SHAFFT_MINIMIZE_NDA,
                             0,
                             MPI_COMM_WORLD);

  void* plan = NULL;
  rc = shafftNDCreate(&plan);
  rc = shafftNDInit(
      plan, 3, commDims, dims, SHAFFT_C2C, MPI_COMM_WORLD, SHAFFT_LAYOUT_REDISTRIBUTED);

  size_t elemCount = 0;
  rc = shafftGetAllocSize(plan, &elemCount);
  size_t localElems = subsize[0] * subsize[1] * subsize[2];

  void*data = NULL, *work = NULL;
  rc = shafftAllocBufferF(elemCount, &data);
  rc = shafftAllocBufferF(elemCount, &work);

  float* host = (float*)calloc(elemCount * 2, sizeof(float));
  if (rank == 0 && localElems > 0)
    host[0] = 1.0f;

  rc = shafftCopyToBufferF(data, host, elemCount);
  rc = shafftSetBuffers(plan, data, work);
  rc = shafftPlan(plan);

  rc = shafftExecute(plan, SHAFFT_FORWARD);
  rc = shafftNormalize(plan);
  rc = shafftGetBuffers(plan, &data, &work);

  float* spectrum = (float*)malloc(elemCount * 2 * sizeof(float));
  rc = shafftCopyFromBufferF(spectrum, data, elemCount);

  if (rank == 0) {
    printf("Spectrum[0..3] = (%g,%g) (%g,%g) (%g,%g) (%g,%g)\n",
           spectrum[0],
           spectrum[1],
           spectrum[2],
           spectrum[3],
           spectrum[4],
           spectrum[5],
           spectrum[6],
           spectrum[7]);
  }

  rc = shafftExecute(plan, SHAFFT_BACKWARD);
  rc = shafftNormalize(plan);
  rc = shafftGetBuffers(plan, &data, &work);

  float* result = (float*)malloc(elemCount * 2 * sizeof(float));
  rc = shafftCopyFromBufferF(result, data, elemCount);

  if (rank == 0) {
    printf("Result[0..3]   = (%g,%g) (%g,%g) (%g,%g) (%g,%g)\n",
           result[0],
           result[1],
           result[2],
           result[3],
           result[4],
           result[5],
           result[6],
           result[7]);
  }

  free(result);
  free(spectrum);
  free(host);
  rc = shafftFreeBufferF(data);
  rc = shafftFreeBufferF(work);
  rc = shafftDestroy(&plan);

  MPI_Finalize();
  return 0;
}
