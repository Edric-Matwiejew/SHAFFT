/**
 * @file test_c_FFT1D.c
 * @brief Integration tests for the C 1D distributed FFT API
 */
#include <math.h>
#include <mpi.h>
#include <shafft/shafft.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Test result tracking */
static int g_rank = 0;
static int g_size = 1;
static int g_passed = 0;
static int g_failed = 0;

#define TEST_BEGIN(name)                                                                           \
  do {                                                                                             \
    if (g_rank == 0)                                                                               \
      printf("  %-40s ", name);                                                                    \
  } while (0)

#define TEST_PASS()                                                                                \
  do {                                                                                             \
    if (g_rank == 0)                                                                               \
      printf("PASS\n");                                                                            \
    g_passed++;                                                                                    \
    return;                                                                                        \
  } while (0)

#define TEST_FAIL(msg)                                                                             \
  do {                                                                                             \
    if (g_rank == 0)                                                                               \
      printf("FAIL (%s)\n", msg);                                                                  \
    g_failed++;                                                                                    \
    return;                                                                                        \
  } while (0)

#define TEST_FAIL_RC(msg, rc)                                                                      \
  do {                                                                                             \
    if (g_rank == 0) {                                                                             \
      char _buf[256] = {0};                                                                        \
      shafftLastErrorMessage(_buf, sizeof(_buf));                                                  \
      printf("FAIL (%s rc=%d err=\"%s\")\n", msg, rc, _buf);                                       \
    }                                                                                              \
    g_failed++;                                                                                    \
    return;                                                                                        \
  } while (0)

/* Single-precision complex type */
typedef struct {
  float real;
  float imag;
} complexf;

/* Double-precision complex type */
typedef struct {
  double real;
  double imag;
} complexd;

/* Tolerance for floating point comparison */
#define TOLERANCE_F 1e-4f
#define TOLERANCE_D 1e-10

/*============================================================================
 * Configuration Tests
 *============================================================================*/

static void test_configuration1d_basic(void) {
  TEST_BEGIN("shafftConfiguration1D basic");

  size_t N = 1024;
  size_t localN = 0, localStart = 0, localAllocSize = 0;

  int rc =
      shafftConfiguration1D(N, &localN, &localStart, &localAllocSize, SHAFFT_C2C, MPI_COMM_WORLD);
  if (rc != 0)
    TEST_FAIL_RC("shafftConfiguration1D", rc);

  /* Verify basic sanity */
  if (localAllocSize == 0)
    TEST_FAIL("localAllocSize is 0");

  /* localN should be reasonable */
  if (localN > N)
    TEST_FAIL("localN > N");

  /* localStart should be within bounds */
  if (localStart >= N && localN > 0)
    TEST_FAIL("localStart out of bounds");

  TEST_PASS();
}

static void test_configuration1d_double(void) {
  TEST_BEGIN("shafftConfiguration1D Z2Z");

  size_t N = 2048;
  size_t localN = 0, localStart = 0, localAllocSize = 0;

  int rc =
      shafftConfiguration1D(N, &localN, &localStart, &localAllocSize, SHAFFT_Z2Z, MPI_COMM_WORLD);
  if (rc != 0)
    TEST_FAIL_RC("shafftConfiguration1D", rc);

  if (localAllocSize == 0)
    TEST_FAIL("localAllocSize is 0");

  TEST_PASS();
}

static void test_configuration1d_coverage(void) {
  TEST_BEGIN("shafftConfiguration1D coverage");

  size_t N = 512;
  size_t localN = 0, localStart = 0, localAllocSize = 0;

  int rc =
      shafftConfiguration1D(N, &localN, &localStart, &localAllocSize, SHAFFT_C2C, MPI_COMM_WORLD);
  if (rc != 0)
    TEST_FAIL_RC("shafftConfiguration1D", rc);

  /* Gather all localN and localStart to verify complete coverage */
  size_t* all_local_n = NULL;
  size_t* all_local_start = NULL;

  if (g_rank == 0) {
    all_local_n = (size_t*)malloc((size_t)g_size * sizeof(size_t));
    all_local_start = (size_t*)malloc((size_t)g_size * sizeof(size_t));
  }

  MPI_Gather(&localN, 1, MPI_UNSIGNED_LONG, all_local_n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  MPI_Gather(
      &localStart, 1, MPI_UNSIGNED_LONG, all_local_start, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  int pass = 1;
  if (g_rank == 0) {
    /* Verify total coverage equals N */
    size_t total = 0;
    for (int i = 0; i < g_size; i++) {
      total += all_local_n[i];
    }
    if (total != N)
      pass = 0;

    /* Verify no overlaps (starts are increasing) */
    for (int i = 1; i < g_size; i++) {
      if (all_local_start[i] < all_local_start[i - 1] + all_local_n[i - 1]) {
        pass = 0;
      }
    }

    free(all_local_n);
    free(all_local_start);
  }

  MPI_Bcast(&pass, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (!pass)
    TEST_FAIL("coverage mismatch");

  TEST_PASS();
}

/*============================================================================
 * Plan Lifecycle Tests
 *============================================================================*/

static void test_FFT1D_create_destroy(void) {
  TEST_BEGIN("FFT1D create/destroy");

  void* plan = NULL;
  int rc = shafft1DCreate(&plan);
  if (rc != 0)
    TEST_FAIL_RC("shafft1DCreate", rc);
  if (plan == NULL)
    TEST_FAIL("plan is NULL");

  rc = shafftDestroy(&plan);
  if (rc != 0)
    TEST_FAIL_RC("shafftDestroy", rc);
  if (plan != NULL)
    TEST_FAIL("plan not NULL after destroy");

  TEST_PASS();
}

static void test_FFT1D_init(void) {
  TEST_BEGIN("FFT1D init");

  size_t N = 1024;
  size_t localN = 0, localStart = 0, localAllocSize = 0;

  int rc =
      shafftConfiguration1D(N, &localN, &localStart, &localAllocSize, SHAFFT_C2C, MPI_COMM_WORLD);
  if (rc != 0)
    TEST_FAIL_RC("shafftConfiguration1D", rc);

  void* plan = NULL;
  rc = shafft1DCreate(&plan);
  if (rc != 0)
    TEST_FAIL_RC("shafft1DCreate", rc);

  rc = shafft1DInit(plan, N, localN, localStart, SHAFFT_C2C, MPI_COMM_WORLD);
  if (rc != 0) {
    shafftDestroy(&plan);
    TEST_FAIL_RC("shafft1DInit", rc);
  }

  /* Check configured state */
  int configured = 0;
  rc = shafftIsConfigured(plan, &configured);
  if (rc != 0 || !configured) {
    shafftDestroy(&plan);
    TEST_FAIL("not configured after init");
  }

  shafftDestroy(&plan);
  TEST_PASS();
}

static void test_FFT1D_query_methods(void) {
  TEST_BEGIN("FFT1D query methods");

  size_t N = 2048;
  size_t localN = 0, localStart = 0, localAllocSize = 0;

  int rc =
      shafftConfiguration1D(N, &localN, &localStart, &localAllocSize, SHAFFT_C2C, MPI_COMM_WORLD);
  if (rc != 0)
    TEST_FAIL_RC("shafftConfiguration1D", rc);

  void* plan = NULL;
  shafft1DCreate(&plan);
  shafft1DInit(plan, N, localN, localStart, SHAFFT_C2C, MPI_COMM_WORLD);

  /* Test GetAllocSize */
  size_t plan_alloc = 0;
  rc = shafftGetAllocSize(plan, &plan_alloc);
  if (rc != 0) {
    shafftDestroy(&plan);
    TEST_FAIL_RC("GetAllocSize", rc);
  }
  if (plan_alloc == 0) {
    shafftDestroy(&plan);
    TEST_FAIL("localAllocSize is 0");
  }

  /* Test GetGlobalSize */
  size_t plan_N = 0;
  rc = shafftGetGlobalSize(plan, &plan_N);
  if (rc != 0) {
    shafftDestroy(&plan);
    TEST_FAIL_RC("GetGlobalSize", rc);
  }
  if (plan_N != N) {
    shafftDestroy(&plan);
    TEST_FAIL("N mismatch");
  }

  /* Test GetLayout - for 1D plans, subsize and offset are single-element arrays */
  size_t plan_local_n = 0, plan_local_start = 0;
  rc = shafftGetLayout(plan, &plan_local_n, &plan_local_start, SHAFFT_TENSOR_LAYOUT_INITIAL);
  if (rc != 0) {
    shafftDestroy(&plan);
    TEST_FAIL_RC("GetLayout", rc);
  }
  if (plan_local_n != localN) {
    shafftDestroy(&plan);
    TEST_FAIL("localN mismatch");
  }
  if (plan_local_start != localStart) {
    shafftDestroy(&plan);
    TEST_FAIL("localStart mismatch");
  }

  /* Test IsActive */
  int active = 0;
  rc = shafftIsActive(plan, &active);
  if (rc != 0) {
    shafftDestroy(&plan);
    TEST_FAIL_RC("IsActive", rc);
  }

  shafftDestroy(&plan);
  TEST_PASS();
}

/*============================================================================
 * Roundtrip Tests
 *============================================================================*/

static void test_FFT1D_roundtrip_float(void) {
  TEST_BEGIN("FFT1D roundtrip C2C");

  size_t N = 1024;
  size_t localN = 0, localStart = 0, localAllocSize = 0;

  int rc =
      shafftConfiguration1D(N, &localN, &localStart, &localAllocSize, SHAFFT_C2C, MPI_COMM_WORLD);
  if (rc != 0)
    TEST_FAIL_RC("shafftConfiguration1D", rc);

  void* plan = NULL;
  rc = shafft1DCreate(&plan);
  if (rc != 0)
    TEST_FAIL_RC("shafft1DCreate", rc);

  rc = shafft1DInit(plan, N, localN, localStart, SHAFFT_C2C, MPI_COMM_WORLD);
  if (rc != 0) {
    shafftDestroy(&plan);
    TEST_FAIL_RC("shafft1DInit", rc);
  }

  /* Get allocation size from plan */
  size_t plan_alloc = 0;
  rc = shafftGetAllocSize(plan, &plan_alloc);
  if (rc != 0) {
    shafftDestroy(&plan);
    TEST_FAIL_RC("GetAllocSize", rc);
  }

  /* Allocate buffers */
  void*data = NULL, *work = NULL;
  rc = shafftAllocBufferF(plan_alloc, &data);
  if (rc != 0) {
    shafftDestroy(&plan);
    TEST_FAIL_RC("AllocBufferF(data)", rc);
  }
  rc = shafftAllocBufferF(plan_alloc, &work);
  if (rc != 0) {
    (void)shafftFreeBufferF(data);
    shafftDestroy(&plan);
    TEST_FAIL_RC("AllocBufferF(work)", rc);
  }

  /* Create host data */
  complexf* host_orig = (complexf*)malloc(plan_alloc * sizeof(complexf));
  complexf* host_result = (complexf*)malloc(plan_alloc * sizeof(complexf));
  if (!host_orig || !host_result) {
    (void)shafftFreeBufferF(data);
    (void)shafftFreeBufferF(work);
    shafftDestroy(&plan);
    free(host_orig);
    free(host_result);
    TEST_FAIL("host alloc failed");
  }

  /* Initialize with pattern */
  for (size_t i = 0; i < plan_alloc; i++) {
    host_orig[i].real = (float)(i % 100) / 100.0f;
    host_orig[i].imag = (float)((i + 50) % 100) / 100.0f;
  }

  /* Copy to device */
  rc = shafftCopyToBufferF(data, host_orig, plan_alloc);
  if (rc != 0) {
    (void)shafftFreeBufferF(data);
    (void)shafftFreeBufferF(work);
    shafftDestroy(&plan);
    free(host_orig);
    free(host_result);
    TEST_FAIL_RC("CopyToBufferF", rc);
  }

  /* Set buffers */
  rc = shafftSetBuffers(plan, data, work);
  if (rc != 0) {
    (void)shafftFreeBufferF(data);
    (void)shafftFreeBufferF(work);
    shafftDestroy(&plan);
    free(host_orig);
    free(host_result);
    TEST_FAIL_RC("SetBuffers", rc);
  }

  rc = shafftPlan(plan);
  if (rc != 0) {
    (void)shafftFreeBufferF(data);
    (void)shafftFreeBufferF(work);
    shafftDestroy(&plan);
    free(host_orig);
    free(host_result);
    TEST_FAIL_RC("Plan", rc);
  }

  /* Forward transform */
  rc = shafftExecute(plan, SHAFFT_FORWARD);
  if (rc != 0) {
    (void)shafftFreeBufferF(data);
    (void)shafftFreeBufferF(work);
    shafftDestroy(&plan);
    free(host_orig);
    free(host_result);
    TEST_FAIL_RC("Execute(FORWARD)", rc);
  }

  /* Backward transform */
  rc = shafftExecute(plan, SHAFFT_BACKWARD);
  if (rc != 0) {
    (void)shafftFreeBufferF(data);
    (void)shafftFreeBufferF(work);
    shafftDestroy(&plan);
    free(host_orig);
    free(host_result);
    TEST_FAIL_RC("Execute(BACKWARD)", rc);
  }

  /* Normalize */
  rc = shafftNormalize(plan);
  if (rc != 0) {
    (void)shafftFreeBufferF(data);
    (void)shafftFreeBufferF(work);
    shafftDestroy(&plan);
    free(host_orig);
    free(host_result);
    TEST_FAIL_RC("Normalize", rc);
  }

  /* Get result buffer */
  void*result_data = NULL, *result_work = NULL;
  rc = shafftGetBuffers(plan, &result_data, &result_work);
  if (rc != 0) {
    (void)shafftFreeBufferF(data);
    (void)shafftFreeBufferF(work);
    shafftDestroy(&plan);
    free(host_orig);
    free(host_result);
    TEST_FAIL_RC("GetBuffers", rc);
  }

  /* Copy back */
  rc = shafftCopyFromBufferF(host_result, result_data, plan_alloc);
  if (rc != 0) {
    (void)shafftFreeBufferF(data);
    (void)shafftFreeBufferF(work);
    shafftDestroy(&plan);
    free(host_orig);
    free(host_result);
    TEST_FAIL_RC("CopyFromBufferF", rc);
  }

  /* Compare */
  float max_err = 0.0f;
  for (size_t i = 0; i < localN; i++) {
    float err_real = fabsf(host_result[i].real - host_orig[i].real);
    float err_imag = fabsf(host_result[i].imag - host_orig[i].imag);
    if (err_real > max_err)
      max_err = err_real;
    if (err_imag > max_err)
      max_err = err_imag;
  }

  int pass = (max_err < TOLERANCE_F);

  /* Cleanup */
  (void)shafftFreeBufferF(data);
  (void)shafftFreeBufferF(work);
  shafftDestroy(&plan);
  free(host_orig);
  free(host_result);

  if (!pass)
    TEST_FAIL("roundtrip error too large");

  TEST_PASS();
}

static void test_FFT1D_roundtrip_double(void) {
  TEST_BEGIN("FFT1D roundtrip Z2Z");

  size_t N = 2048;
  size_t localN = 0, localStart = 0, localAllocSize = 0;

  int rc =
      shafftConfiguration1D(N, &localN, &localStart, &localAllocSize, SHAFFT_Z2Z, MPI_COMM_WORLD);
  if (rc != 0)
    TEST_FAIL_RC("shafftConfiguration1D", rc);

  void* plan = NULL;
  rc = shafft1DCreate(&plan);
  if (rc != 0)
    TEST_FAIL_RC("shafft1DCreate", rc);

  rc = shafft1DInit(plan, N, localN, localStart, SHAFFT_Z2Z, MPI_COMM_WORLD);
  if (rc != 0) {
    shafftDestroy(&plan);
    TEST_FAIL_RC("shafft1DInit", rc);
  }

  /* Get allocation size */
  size_t plan_alloc = 0;
  rc = shafftGetAllocSize(plan, &plan_alloc);
  if (rc != 0) {
    shafftDestroy(&plan);
    TEST_FAIL_RC("GetAllocSize", rc);
  }

  /* Allocate buffers */
  void*data = NULL, *work = NULL;
  rc = shafftAllocBufferD(plan_alloc, &data);
  if (rc != 0) {
    shafftDestroy(&plan);
    TEST_FAIL_RC("AllocBufferD(data)", rc);
  }
  rc = shafftAllocBufferD(plan_alloc, &work);
  if (rc != 0) {
    (void)shafftFreeBufferD(data);
    shafftDestroy(&plan);
    TEST_FAIL_RC("AllocBufferD(work)", rc);
  }

  /* Create host data */
  complexd* host_orig = (complexd*)malloc(plan_alloc * sizeof(complexd));
  complexd* host_result = (complexd*)malloc(plan_alloc * sizeof(complexd));
  if (!host_orig || !host_result) {
    (void)shafftFreeBufferD(data);
    (void)shafftFreeBufferD(work);
    shafftDestroy(&plan);
    free(host_orig);
    free(host_result);
    TEST_FAIL("host alloc failed");
  }

  /* Initialize with pattern */
  for (size_t i = 0; i < plan_alloc; i++) {
    host_orig[i].real = (double)(i % 100) / 100.0;
    host_orig[i].imag = (double)((i + 50) % 100) / 100.0;
  }

  /* Copy to device */
  rc = shafftCopyToBufferD(data, host_orig, plan_alloc);
  if (rc != 0) {
    (void)shafftFreeBufferD(data);
    (void)shafftFreeBufferD(work);
    shafftDestroy(&plan);
    free(host_orig);
    free(host_result);
    TEST_FAIL_RC("CopyToBufferD", rc);
  }

  /* Set buffers */
  rc = shafftSetBuffers(plan, data, work);
  if (rc != 0) {
    (void)shafftFreeBufferD(data);
    (void)shafftFreeBufferD(work);
    shafftDestroy(&plan);
    free(host_orig);
    free(host_result);
    TEST_FAIL_RC("SetBuffers", rc);
  }

  rc = shafftPlan(plan);
  if (rc != 0) {
    (void)shafftFreeBufferD(data);
    (void)shafftFreeBufferD(work);
    shafftDestroy(&plan);
    free(host_orig);
    free(host_result);
    TEST_FAIL_RC("Plan", rc);
  }

  /* Forward -> Backward -> Normalize */
  rc = shafftExecute(plan, SHAFFT_FORWARD);
  if (rc != 0) {
    (void)shafftFreeBufferD(data);
    (void)shafftFreeBufferD(work);
    shafftDestroy(&plan);
    free(host_orig);
    free(host_result);
    TEST_FAIL_RC("Execute(FORWARD)", rc);
  }

  rc = shafftExecute(plan, SHAFFT_BACKWARD);
  if (rc != 0) {
    (void)shafftFreeBufferD(data);
    (void)shafftFreeBufferD(work);
    shafftDestroy(&plan);
    free(host_orig);
    free(host_result);
    TEST_FAIL_RC("Execute(BACKWARD)", rc);
  }

  rc = shafftNormalize(plan);
  if (rc != 0) {
    (void)shafftFreeBufferD(data);
    (void)shafftFreeBufferD(work);
    shafftDestroy(&plan);
    free(host_orig);
    free(host_result);
    TEST_FAIL_RC("Normalize", rc);
  }

  /* Get result buffer */
  void*result_data = NULL, *result_work = NULL;
  rc = shafftGetBuffers(plan, &result_data, &result_work);
  if (rc != 0) {
    (void)shafftFreeBufferD(data);
    (void)shafftFreeBufferD(work);
    shafftDestroy(&plan);
    free(host_orig);
    free(host_result);
    TEST_FAIL_RC("GetBuffers", rc);
  }

  /* Copy back */
  rc = shafftCopyFromBufferD(host_result, result_data, plan_alloc);
  if (rc != 0) {
    (void)shafftFreeBufferD(data);
    (void)shafftFreeBufferD(work);
    shafftDestroy(&plan);
    free(host_orig);
    free(host_result);
    TEST_FAIL_RC("CopyFromBufferD", rc);
  }

  /* Compare */
  double max_err = 0.0;
  for (size_t i = 0; i < localN; i++) {
    double err_real = fabs(host_result[i].real - host_orig[i].real);
    double err_imag = fabs(host_result[i].imag - host_orig[i].imag);
    if (err_real > max_err)
      max_err = err_real;
    if (err_imag > max_err)
      max_err = err_imag;
  }

  int pass = (max_err < TOLERANCE_D);

  /* Cleanup */
  (void)shafftFreeBufferD(data);
  (void)shafftFreeBufferD(work);
  shafftDestroy(&plan);
  free(host_orig);
  free(host_result);

  if (!pass)
    TEST_FAIL("roundtrip error too large");

  TEST_PASS();
}

/*============================================================================
 * Delta Function Test
 *============================================================================*/

static void test_FFT1D_delta(void) {
  TEST_BEGIN("FFT1D delta function");

  size_t N = 512;
  size_t localN = 0, localStart = 0, localAllocSize = 0;

  int rc =
      shafftConfiguration1D(N, &localN, &localStart, &localAllocSize, SHAFFT_C2C, MPI_COMM_WORLD);
  if (rc != 0)
    TEST_FAIL_RC("shafftConfiguration1D", rc);

  void* plan = NULL;
  shafft1DCreate(&plan);
  shafft1DInit(plan, N, localN, localStart, SHAFFT_C2C, MPI_COMM_WORLD);

  size_t plan_alloc = 0;
  shafftGetAllocSize(plan, &plan_alloc);

  void*data = NULL, *work = NULL;
  shafftAllocBufferF(plan_alloc, &data);
  shafftAllocBufferF(plan_alloc, &work);

  complexf* host_data = (complexf*)calloc(plan_alloc, sizeof(complexf));
  if (!host_data) {
    (void)shafftFreeBufferF(data);
    (void)shafftFreeBufferF(work);
    shafftDestroy(&plan);
    TEST_FAIL("host alloc failed");
  }

  /* Set delta at index 0 (only rank 0 has it) */
  if (localStart == 0 && localN > 0) {
    host_data[0].real = 1.0f;
    host_data[0].imag = 0.0f;
  }

  rc = shafftCopyToBufferF(data, host_data, plan_alloc);
  if (rc != 0) {
    (void)shafftFreeBufferF(data);
    (void)shafftFreeBufferF(work);
    shafftDestroy(&plan);
    free(host_data);
    TEST_FAIL_RC("CopyToBufferF", rc);
  }

  rc = shafftSetBuffers(plan, data, work);
  if (rc != 0) {
    (void)shafftFreeBufferF(data);
    (void)shafftFreeBufferF(work);
    shafftDestroy(&plan);
    free(host_data);
    TEST_FAIL_RC("SetBuffers", rc);
  }

  rc = shafftPlan(plan);
  if (rc != 0) {
    (void)shafftFreeBufferF(data);
    (void)shafftFreeBufferF(work);
    shafftDestroy(&plan);
    free(host_data);
    TEST_FAIL_RC("Plan", rc);
  }

  /* Forward transform */
  rc = shafftExecute(plan, SHAFFT_FORWARD);
  if (rc != 0) {
    (void)shafftFreeBufferF(data);
    (void)shafftFreeBufferF(work);
    shafftDestroy(&plan);
    free(host_data);
    TEST_FAIL_RC("Execute(FORWARD)", rc);
  }

  /* Copy from work buffer (output after forward FFT) */
  rc = shafftCopyFromBufferF(host_data, work, plan_alloc);
  if (rc != 0) {
    (void)shafftFreeBufferF(data);
    (void)shafftFreeBufferF(work);
    shafftDestroy(&plan);
    free(host_data);
    TEST_FAIL_RC("CopyFromBufferF", rc);
  }

  /* FFT of delta should be constant: real=1.0, imag=0.0 */
  float max_err = 0.0f;
  for (size_t i = 0; i < localN; i++) {
    float err_real = fabsf(host_data[i].real - 1.0f);
    float err_imag = fabsf(host_data[i].imag);
    if (err_real > max_err)
      max_err = err_real;
    if (err_imag > max_err)
      max_err = err_imag;
  }

  int pass = (max_err < TOLERANCE_F);

  (void)shafftFreeBufferF(data);
  (void)shafftFreeBufferF(work);
  shafftDestroy(&plan);
  free(host_data);

  if (!pass)
    TEST_FAIL("delta spectrum not constant 1.0");

  TEST_PASS();
}

/*============================================================================
 * Buffer Swap Test
 *============================================================================*/

static void test_FFT1D_buffer_swap(void) {
  TEST_BEGIN("FFT1D buffer swap tracking");

  size_t N = 256;
  size_t localN = 0, localStart = 0, localAllocSize = 0;

  shafftConfiguration1D(N, &localN, &localStart, &localAllocSize, SHAFFT_C2C, MPI_COMM_WORLD);

  void* plan = NULL;
  shafft1DCreate(&plan);
  shafft1DInit(plan, N, localN, localStart, SHAFFT_C2C, MPI_COMM_WORLD);

  size_t plan_alloc = 0;
  shafftGetAllocSize(plan, &plan_alloc);

  void*data = NULL, *work = NULL;
  shafftAllocBufferF(plan_alloc, &data);
  shafftAllocBufferF(plan_alloc, &work);

  shafftSetBuffers(plan, data, work);
  int rc = shafftPlan(plan);
  if (rc != 0) {
    (void)shafftFreeBufferF(data);
    (void)shafftFreeBufferF(work);
    shafftDestroy(&plan);
    TEST_FAIL_RC("Plan", rc);
  }

  /* Get initial buffers */
  void*buf1_data = NULL, *buf1_work = NULL;
  shafftGetBuffers(plan, &buf1_data, &buf1_work);

  if (buf1_data != data || buf1_work != work) {
    (void)shafftFreeBufferF(data);
    (void)shafftFreeBufferF(work);
    shafftDestroy(&plan);
    TEST_FAIL("initial buffer mismatch");
  }

  /* Execute forward (may swap buffers) */
  rc = shafftExecute(plan, SHAFFT_FORWARD);
  if (rc != 0) {
    (void)shafftFreeBufferF(data);
    (void)shafftFreeBufferF(work);
    shafftDestroy(&plan);
    TEST_FAIL_RC("Execute(FORWARD)", rc);
  }

  /* Get buffers after - they should be valid (though may be swapped) */
  void*buf2_data = NULL, *buf2_work = NULL;
  shafftGetBuffers(plan, &buf2_data, &buf2_work);

  /* Both should be one of {data, work} */
  int valid = ((buf2_data == data || buf2_data == work) &&
               (buf2_work == data || buf2_work == work) && buf2_data != buf2_work);

  (void)shafftFreeBufferF(data);
  (void)shafftFreeBufferF(work);
  shafftDestroy(&plan);

  if (!valid)
    TEST_FAIL("invalid buffer state after execute");

  TEST_PASS();
}

/*============================================================================
 * Main
 *============================================================================*/

int main(int argc, char** argv) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    fprintf(stderr, "MPI_Init failed\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_size);

  if (g_rank == 0) {
    printf("\n=== C API FFT1D Integration Tests ===\n");
    printf("Backend: %s\n", shafftGetBackendName());
    printf("MPI ranks: %d\n\n", g_size);
  }

  /* Configuration tests */
  test_configuration1d_basic();
  test_configuration1d_double();
  test_configuration1d_coverage();

  /* Plan lifecycle tests */
  test_FFT1D_create_destroy();
  test_FFT1D_init();
  test_FFT1D_query_methods();

  /* Functional tests */
  test_FFT1D_roundtrip_float();
  test_FFT1D_roundtrip_double();
  test_FFT1D_delta();
  test_FFT1D_buffer_swap();

  /* Summary */
  if (g_rank == 0) {
    printf("\n----------------------------------------\n");
    printf("Passed: %d  Failed: %d\n", g_passed, g_failed);
    printf("----------------------------------------\n\n");
  }

  MPI_Finalize();
  return (g_failed > 0) ? 1 : 0;
}
