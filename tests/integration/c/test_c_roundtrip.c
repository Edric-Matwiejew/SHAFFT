/**
 * @file test_c_roundtrip.c
 * @brief Test that the C API can perform a basic forward/backward/normalize roundtrip
 */
#include "test_utils.h"
#include <math.h>
#include <mpi.h>
#include <shafft/shafft.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FAIL_WITH_ERR(label, rc)                                                                   \
  do {                                                                                             \
    char buf[256] = {0};                                                                           \
    shafftLastErrorMessage(buf, sizeof(buf));                                                      \
    fprintf(stderr, "%s failed rc=%d err=\"%s\"\n", label, rc, buf);                               \
    return 1;                                                                                      \
  } while (0)

static int test_roundtrip_basic(void) {
  int dims[3] = {32, 32, 32};
  const int ndim = 3;
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  /* For nda=1 decomposition: distribute on first axis only */
  int commDims[3] = {worldSize, 1, 1};
  int rc;

  /* Create plan */
  void* plan = NULL;
  rc = shafftNDCreate(&plan);
  if (rc != 0 || plan == NULL)
    FAIL_WITH_ERR("shafftNDCreate", rc);

  /* Initialize plan */
  rc = shafftNDInit(
      plan, ndim, commDims, dims, SHAFFT_C2C, MPI_COMM_WORLD, SHAFFT_LAYOUT_REDISTRIBUTED);
  if (rc != 0) {
    shafftDestroy(&plan);
    FAIL_WITH_ERR("shafftNDInit", rc);
  }

  /* Get allocation size */
  size_t localAllocSize = 0;
  rc = shafftGetAllocSize(plan, &localAllocSize);
  if (rc != 0 || localAllocSize == 0) {
    shafftDestroy(&plan);
    FAIL_WITH_ERR("shafftGetAllocSize", rc);
  }

  /* Allocate device buffers */
  void*data = NULL, *work = NULL;
  rc = shafftAllocBufferF(localAllocSize, &data);
  if (rc != 0) {
    shafftDestroy(&plan);
    FAIL_WITH_ERR("shafftAllocBufferF(data)", rc);
  }
  rc = shafftAllocBufferF(localAllocSize, &work);
  if (rc != 0) {
    (void)shafftFreeBufferF(data);
    shafftDestroy(&plan);
    FAIL_WITH_ERR("shafftAllocBufferF(work)", rc);
  }

  /* Create host data with a known pattern */
  test_complexf* host_orig = (test_complexf*)malloc(localAllocSize * sizeof(test_complexf));
  test_complexf* host_result = (test_complexf*)malloc(localAllocSize * sizeof(test_complexf));
  if (!host_orig || !host_result) {
    fprintf(stderr, "Host allocation failed\n");
    (void)shafftFreeBufferF(data);
    (void)shafftFreeBufferF(work);
    shafftDestroy(&plan);
    free(host_orig);
    free(host_result);
    return 1;
  }

  /* Initialize with a pattern */
  for (size_t i = 0; i < localAllocSize; i++) {
    host_orig[i].real = (float)(i % 100) / 100.0f;
    host_orig[i].imag = (float)((i + 50) % 100) / 100.0f;
  }

  /* Copy to device */
  rc = shafftCopyToBufferF(data, host_orig, localAllocSize);
  if (rc != 0) {
    (void)shafftFreeBufferF(data);
    (void)shafftFreeBufferF(work);
    shafftDestroy(&plan);
    free(host_orig);
    free(host_result);
    FAIL_WITH_ERR("shafftCopyToBufferF", rc);
  }

  /* Set buffers */
  rc = shafftSetBuffers(plan, data, work);
  shafftPlan(plan);
  if (rc != 0) {
    (void)shafftFreeBufferF(data);
    (void)shafftFreeBufferF(work);
    shafftDestroy(&plan);
    free(host_orig);
    free(host_result);
    FAIL_WITH_ERR("shafftSetBuffers", rc);
    shafftPlan(plan);
  }

  /* Execute forward transform */
  rc = shafftExecute(plan, SHAFFT_FORWARD);
  if (rc != 0) {
    (void)shafftFreeBufferF(data);
    (void)shafftFreeBufferF(work);
    shafftDestroy(&plan);
    free(host_orig);
    free(host_result);
    FAIL_WITH_ERR("shafftExecute(FORWARD)", rc);
  }

  /* Execute backward transform */
  rc = shafftExecute(plan, SHAFFT_BACKWARD);
  if (rc != 0) {
    (void)shafftFreeBufferF(data);
    (void)shafftFreeBufferF(work);
    shafftDestroy(&plan);
    free(host_orig);
    free(host_result);
    FAIL_WITH_ERR("shafftExecute(BACKWARD)", rc);
  }

  /* Normalize */
  rc = shafftNormalize(plan);
  if (rc != 0) {
    (void)shafftFreeBufferF(data);
    (void)shafftFreeBufferF(work);
    shafftDestroy(&plan);
    free(host_orig);
    free(host_result);
    FAIL_WITH_ERR("shafftNormalize", rc);
  }

  /* Get the result buffer (may have swapped) */
  void*result_data = NULL, *result_work = NULL;
  rc = shafftGetBuffers(plan, &result_data, &result_work);
  if (rc != 0) {
    (void)shafftFreeBufferF(data);
    (void)shafftFreeBufferF(work);
    shafftDestroy(&plan);
    free(host_orig);
    free(host_result);
    FAIL_WITH_ERR("shafftGetBuffers", rc);
  }

  /* Copy result back to host */
  rc = shafftCopyFromBufferF(host_result, result_data, localAllocSize);
  if (rc != 0) {
    (void)shafftFreeBufferF(data);
    (void)shafftFreeBufferF(work);
    shafftDestroy(&plan);
    free(host_orig);
    free(host_result);
    FAIL_WITH_ERR("shafftCopyFromBufferF", rc);
  }

  /* Compare with original using FFTW-style relative error with MPI sync */
  size_t globalN = product_i(dims, ndim);
  int pass = check_rel_error_f((const test_complexf*)host_result,
                               (const test_complexf*)host_orig,
                               localAllocSize,
                               globalN,
                               MPI_COMM_WORLD,
                               TOL_F);
  pass = all_ranks_pass_c(pass, MPI_COMM_WORLD);

  /* Cleanup */
  (void)shafftFreeBufferF(data);
  (void)shafftFreeBufferF(work);
  shafftDestroy(&plan);
  free(host_orig);
  free(host_result);

  if (!pass) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
      fprintf(stderr, "Roundtrip relative error exceeds tolerance\n");
    return 1;
  }

  return 0;
}

static int test_roundtrip_double(void) {
  int dims[2] = {64, 64};
  const int ndim = 2;
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  /* For nda=1 decomposition: distribute on first axis only */
  int commDims[2] = {worldSize, 1};
  int rc;

  /* Create plan */
  void* plan = NULL;
  rc = shafftNDCreate(&plan);
  if (rc != 0)
    return 1;

  rc = shafftNDInit(
      plan, ndim, commDims, dims, SHAFFT_Z2Z, MPI_COMM_WORLD, SHAFFT_LAYOUT_REDISTRIBUTED);
  if (rc != 0) {
    shafftDestroy(&plan);
    return 1;
  }

  size_t localAllocSize = 0;
  rc = shafftGetAllocSize(plan, &localAllocSize);
  if (rc != 0) {
    shafftDestroy(&plan);
    return 1;
  }

  void*data = NULL, *work = NULL;
  rc = shafftAllocBufferD(localAllocSize, &data);
  if (rc != 0) {
    shafftDestroy(&plan);
    return 1;
  }
  rc = shafftAllocBufferD(localAllocSize, &work);
  if (rc != 0) {
    (void)shafftFreeBufferD(data);
    shafftDestroy(&plan);
    return 1;
  }

  test_complexd* host_orig = (test_complexd*)malloc(localAllocSize * sizeof(test_complexd));
  test_complexd* host_result = (test_complexd*)malloc(localAllocSize * sizeof(test_complexd));
  if (!host_orig || !host_result) {
    fprintf(stderr, "Host allocation failed\n");
    (void)shafftFreeBufferD(data);
    (void)shafftFreeBufferD(work);
    shafftDestroy(&plan);
    free(host_orig);
    free(host_result);
    return 1;
  }

  for (size_t i = 0; i < localAllocSize; i++) {
    host_orig[i].real = (double)(i % 100) / 100.0;
    host_orig[i].imag = (double)((i + 50) % 100) / 100.0;
  }

  rc = shafftCopyToBufferD(data, host_orig, localAllocSize);
  if (rc != 0) {
    (void)shafftFreeBufferD(data);
    (void)shafftFreeBufferD(work);
    shafftDestroy(&plan);
    free(host_orig);
    free(host_result);
    FAIL_WITH_ERR("shafftCopyToBufferD", rc);
  }

  rc = shafftSetBuffers(plan, data, work);
  if (rc != 0) {
    (void)shafftFreeBufferD(data);
    (void)shafftFreeBufferD(work);
    shafftDestroy(&plan);
    free(host_orig);
    free(host_result);
    FAIL_WITH_ERR("shafftSetBuffers", rc);
  }

  rc = shafftPlan(plan);
  if (rc != 0) {
    (void)shafftFreeBufferD(data);
    (void)shafftFreeBufferD(work);
    shafftDestroy(&plan);
    free(host_orig);
    free(host_result);
    FAIL_WITH_ERR("shafftPlan", rc);
  }

  rc = shafftExecute(plan, SHAFFT_FORWARD);
  if (rc != 0) {
    (void)shafftFreeBufferD(data);
    (void)shafftFreeBufferD(work);
    shafftDestroy(&plan);
    free(host_orig);
    free(host_result);
    FAIL_WITH_ERR("shafftExecute(FORWARD)", rc);
  }

  rc = shafftExecute(plan, SHAFFT_BACKWARD);
  if (rc != 0) {
    (void)shafftFreeBufferD(data);
    (void)shafftFreeBufferD(work);
    shafftDestroy(&plan);
    free(host_orig);
    free(host_result);
    FAIL_WITH_ERR("shafftExecute(BACKWARD)", rc);
  }

  rc = shafftNormalize(plan);
  if (rc != 0) {
    (void)shafftFreeBufferD(data);
    (void)shafftFreeBufferD(work);
    shafftDestroy(&plan);
    free(host_orig);
    free(host_result);
    FAIL_WITH_ERR("shafftNormalize", rc);
  }

  void*result_data = NULL, *result_work = NULL;
  rc = shafftGetBuffers(plan, &result_data, &result_work);
  if (rc != 0) {
    (void)shafftFreeBufferD(data);
    (void)shafftFreeBufferD(work);
    shafftDestroy(&plan);
    free(host_orig);
    free(host_result);
    FAIL_WITH_ERR("shafftGetBuffers", rc);
  }

  rc = shafftCopyFromBufferD(host_result, result_data, localAllocSize);
  if (rc != 0) {
    (void)shafftFreeBufferD(data);
    (void)shafftFreeBufferD(work);
    shafftDestroy(&plan);
    free(host_orig);
    free(host_result);
    FAIL_WITH_ERR("shafftCopyFromBufferD", rc);
  }

  /* Compare with original using FFTW-style relative error with MPI sync */
  size_t globalN = product_i(dims, ndim);
  int pass =
      check_rel_error_d(host_result, host_orig, localAllocSize, globalN, MPI_COMM_WORLD, TOL_D);
  pass = all_ranks_pass_c(pass, MPI_COMM_WORLD);

  (void)shafftFreeBufferD(data);
  (void)shafftFreeBufferD(work);
  shafftDestroy(&plan);
  free(host_orig);
  free(host_result);

  if (!pass) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
      fprintf(stderr, "Double precision roundtrip relative error exceeds tolerance\n");
    return 1;
  }

  return 0;
}

int main(int argc, char** argv) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    fprintf(stderr, "MPI_Init failed\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int failed = 0;
  int total = 0;

  if (rank == 0) {
    printf("=== C API Roundtrip Tests ===\n");
    printf("SHAFFT %s (backend: %s)\n", shafftGetVersionString(), shafftGetBackendName());
    int worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    printf("MPI ranks: %d\n\n", worldSize);
  }

  /* Test 1: Single precision roundtrip */
  total++;
  if (rank == 0)
    printf("  roundtrip_basic                          ");
  if (test_roundtrip_basic() == 0) {
    if (rank == 0)
      printf("PASS\n");
  } else {
    if (rank == 0)
      printf("FAIL\n");
    failed++;
  }

  /* Test 2: Double precision roundtrip */
  total++;
  if (rank == 0)
    printf("  roundtrip_double                         ");
  if (test_roundtrip_double() == 0) {
    if (rank == 0)
      printf("PASS\n");
  } else {
    if (rank == 0)
      printf("FAIL\n");
    failed++;
  }

  if (rank == 0) {
    printf("\n=== C API Roundtrip Tests: %s ===\n", failed == 0 ? "PASSED" : "FAILED");
  }

  MPI_Finalize();
  return failed > 0 ? 1 : 0;
}
