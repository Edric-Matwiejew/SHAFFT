/**
 * @file test_c_api.c
 * @brief Comprehensive C API coverage test
 */
#include <mpi.h>
#include <shafft/shafft.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Test result tracking */
static int g_rank = 0;
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

#define TEST_SKIP(msg)                                                                             \
  do {                                                                                             \
    if (g_rank == 0)                                                                               \
      printf("SKIP (%s)\n", msg);                                                                  \
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

/*============================================================================
 * Library Info Tests
 *============================================================================*/

static void test_backend_name(void) {
  TEST_BEGIN("shafftGetBackendName");

  const char* backend = shafftGetBackendName();
  if (backend == NULL)
    TEST_FAIL("returned NULL");
  if (strlen(backend) == 0)
    TEST_FAIL("empty string");

  /* Should be "FFTW" or "hipFFT" */
  if (strcmp(backend, "FFTW") != 0 && strcmp(backend, "hipFFT") != 0) {
    TEST_FAIL("unknown backend");
  }

  TEST_PASS();
}

static void test_version(void) {
  TEST_BEGIN("shafftGetVersion");

  int major = -1, minor = -1, patch = -1;
  shafftGetVersion(&major, &minor, &patch);

  if (major < 0 || minor < 0 || patch < 0)
    TEST_FAIL("negative version");

  TEST_PASS();
}

static void test_version_string(void) {
  TEST_BEGIN("shafftGetVersionString");

  const char* ver = shafftGetVersionString();
  if (ver == NULL)
    TEST_FAIL("returned NULL");
  if (strlen(ver) == 0)
    TEST_FAIL("empty string");

  /* Should contain at least one dot */
  if (strchr(ver, '.') == NULL)
    TEST_FAIL("no dots in version");

  TEST_PASS();
}

/*============================================================================
 * Error API Tests
 *============================================================================*/

static void test_error_after_success(void) {
  TEST_BEGIN("error_after_success");

  /* Clear any prior error */
  shafftClearLastError();

  /* Successful operation */
  const char* backend = shafftGetBackendName();
  (void)backend;

  /* Should have no error */
  if (shafftLastErrorStatus() != SHAFFT_SUCCESS)
    TEST_FAIL("expected SUCCESS");

  TEST_PASS();
}

static void test_error_source_name(void) {
  TEST_BEGIN("shafftErrorSourceName");

  const char* name;

  name = shafftErrorSourceName(SHAFFT_ERRSRC_NONE);
  if (name == NULL || strlen(name) == 0)
    TEST_FAIL("NONE name empty");

  name = shafftErrorSourceName(SHAFFT_ERRSRC_MPI);
  if (name == NULL || strlen(name) == 0)
    TEST_FAIL("MPI name empty");

  name = shafftErrorSourceName(SHAFFT_ERRSRC_HIP);
  if (name == NULL || strlen(name) == 0)
    TEST_FAIL("HIP name empty");

  TEST_PASS();
}

static void test_error_message(void) {
  TEST_BEGIN("shafftLastErrorMessage");

  char buf[256] = {0};
  int len = shafftLastErrorMessage(buf, sizeof(buf));

  /* Should return some length (may be 0 if no error) */
  if (len < 0)
    TEST_FAIL("negative length");

  TEST_PASS();
}

static void test_clear_error(void) {
  TEST_BEGIN("shafftClearLastError");

  /* Create an error by passing NULL */
  int rc =
      shafftNDInit(NULL, 3, NULL, NULL, SHAFFT_C2C, MPI_COMM_WORLD, SHAFFT_LAYOUT_REDISTRIBUTED);
  (void)rc;

  /* Should have error */
  if (shafftLastErrorStatus() == SHAFFT_SUCCESS) {
    /* Some implementations may not set error for NULL plan */
    shafftClearLastError();
    TEST_PASS();
  }

  /* Clear it */
  shafftClearLastError();

  /* Should be cleared */
  if (shafftLastErrorStatus() != SHAFFT_SUCCESS)
    TEST_FAIL("not cleared");

  TEST_PASS();
}

/*============================================================================
 * Plan Lifecycle Tests
 *============================================================================*/

static void test_plan_create_destroy(void) {
  TEST_BEGIN("shafftNDCreate/Destroy");

  void* plan = NULL;
  int rc = shafftNDCreate(&plan);
  if (rc != 0)
    TEST_FAIL_RC("create failed", rc);
  if (plan == NULL)
    TEST_FAIL("plan is NULL");

  rc = shafftDestroy(&plan);
  if (rc != 0)
    TEST_FAIL_RC("destroy failed", rc);
  if (plan != NULL)
    TEST_FAIL("plan not set to NULL");

  TEST_PASS();
}

static void test_plan_nd(void) {
  TEST_BEGIN("shafftNDInit");

  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  int dims[3] = {32, 32, 32};
  /* For nda=1: distribute on first axis only */
  int commDims[3] = {worldSize, 1, 1};
  void* plan = NULL;

  int rc = shafftNDCreate(&plan);
  if (rc != 0 || plan == NULL) {
    TEST_FAIL_RC("create failed", rc);
  }
  rc = shafftNDInit(
      plan, 3, commDims, dims, SHAFFT_C2C, MPI_COMM_WORLD, SHAFFT_LAYOUT_REDISTRIBUTED);
  if (rc != 0) {
    shafftDestroy(&plan);
    TEST_FAIL_RC("init failed", rc);
  }

  shafftDestroy(&plan);
  TEST_PASS();
}

static void test_plan_nd_2d(void) {
  TEST_BEGIN("shafftNDInit (2D decomp)");

  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  int dims[3] = {32, 32, 32};
  /* For 2D decomposition, distribute on first two axes */
  /* Use MPI_Dims_create to get a reasonable 2D split */
  int cart_dims[2] = {0, 0};
  MPI_Dims_create(worldSize, 2, cart_dims);
  int commDims[3] = {cart_dims[0], cart_dims[1], 1};
  void* plan = NULL;

  int rc = shafftNDCreate(&plan);
  if (rc != 0 || plan == NULL) {
    TEST_FAIL_RC("create failed", rc);
  }
  rc = shafftNDInit(
      plan, 3, commDims, dims, SHAFFT_C2C, MPI_COMM_WORLD, SHAFFT_LAYOUT_REDISTRIBUTED);
  if (rc != 0) {
    shafftDestroy(&plan);
    TEST_FAIL_RC("init failed", rc);
  }

  shafftDestroy(&plan);
  TEST_PASS();
}

static void test_double_destroy(void) {
  TEST_BEGIN("double_destroy");

  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  int dims[2] = {16, 16};
  int commDims[2] = {worldSize, 1};
  void* plan = NULL;

  int rc = shafftNDCreate(&plan);
  if (rc != 0 || plan == NULL) {
    TEST_FAIL_RC("create failed", rc);
  }
  rc = shafftNDInit(
      plan, 2, commDims, dims, SHAFFT_C2C, MPI_COMM_WORLD, SHAFFT_LAYOUT_REDISTRIBUTED);
  if (rc != 0) {
    shafftDestroy(&plan);
    TEST_FAIL_RC("init failed", rc);
  }

  int rc1 = shafftDestroy(&plan);
  int rc2 = shafftDestroy(&plan); /* Should handle NULL gracefully */

  if (rc1 != 0)
    TEST_FAIL("first destroy failed");
  /* Second destroy on NULL may or may not fail, just shouldn't crash */
  (void)rc2;

  TEST_PASS();
}

static void test_plan_with_init_policy(void) {
  TEST_BEGIN("shafftNDInit(policy)+Plan");

  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  int dims[3] = {32, 32, 32};
  int commDims[3] = {worldSize, 1, 1};
  void* plan = NULL;
  size_t n = 0;
  void* data = NULL;
  void* work = NULL;

  int rc = shafftNDCreate(&plan);
  if (rc != 0 || plan == NULL)
    TEST_FAIL_RC("create failed", rc);

  rc = shafftNDInit(plan, 3, commDims, dims, SHAFFT_C2C, MPI_COMM_WORLD, SHAFFT_LAYOUT_INITIAL);
  if (rc != 0) {
    shafftDestroy(&plan);
    TEST_FAIL_RC("init failed", rc);
  }

  rc = shafftGetAllocSize(plan, &n);
  if (rc != 0 || n == 0) {
    shafftDestroy(&plan);
    TEST_FAIL_RC("alloc size query failed", rc);
  }

  rc = shafftAllocBufferF(n, &data);
  if (rc != 0) {
    shafftDestroy(&plan);
    TEST_FAIL_RC("data alloc failed", rc);
  }
  rc = shafftAllocBufferF(n, &work);
  if (rc != 0) {
    (void)shafftFreeBufferF(data);
    shafftDestroy(&plan);
    TEST_FAIL_RC("work alloc failed", rc);
  }

  rc = shafftSetBuffers(plan, data, work);
  if (rc != 0) {
    (void)shafftFreeBufferF(data);
    (void)shafftFreeBufferF(work);
    shafftDestroy(&plan);
    TEST_FAIL_RC("set buffers failed", rc);
  }

  rc = shafftPlan(plan);
  if (rc != 0) {
    (void)shafftFreeBufferF(data);
    (void)shafftFreeBufferF(work);
    shafftDestroy(&plan);
    TEST_FAIL_RC("plan failed", rc);
  }

  (void)shafftFreeBufferF(data);
  (void)shafftFreeBufferF(work);
  shafftDestroy(&plan);
  TEST_PASS();
}

static void test_plan_invalid_output_policy(void) {
  TEST_BEGIN("shafftNDInit(invalid_policy)");

  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  int dims[3] = {16, 16, 16};
  int commDims[3] = {worldSize, 1, 1};
  void* plan = NULL;

  int rc = shafftNDCreate(&plan);
  if (rc != 0 || plan == NULL)
    TEST_FAIL_RC("create failed", rc);

  rc = shafftNDInit(
      plan, 3, commDims, dims, SHAFFT_C2C, MPI_COMM_WORLD, (shafft_transform_layout_t)2);
  if (rc != SHAFFT_ERR_INVALID_LAYOUT) {
    shafftDestroy(&plan);
    TEST_FAIL_RC("expected INVALID_LAYOUT", rc);
  }

  shafftDestroy(&plan);
  TEST_PASS();
}

/*============================================================================
 * Buffer Management Tests
 *============================================================================*/

static void test_alloc_free_buffer_f(void) {
  TEST_BEGIN("shafftAllocBufferF/FreeBufferF");

  void* buf = NULL;
  int rc = shafftAllocBufferF(1024, &buf);
  if (rc != 0)
    TEST_FAIL("alloc failed");
  if (buf == NULL)
    TEST_FAIL("buf is NULL");

  rc = shafftFreeBufferF(buf);
  if (rc != 0)
    TEST_FAIL("free failed");

  /* Free NULL should be safe */
  rc = shafftFreeBufferF(NULL);
  if (rc != 0)
    TEST_FAIL("free NULL failed");

  TEST_PASS();
}

static void test_alloc_free_buffer_d(void) {
  TEST_BEGIN("shafftAllocBufferD/FreeBufferD");

  void* buf = NULL;
  int rc = shafftAllocBufferD(1024, &buf);
  if (rc != 0)
    TEST_FAIL("alloc failed");
  if (buf == NULL)
    TEST_FAIL("buf is NULL");

  rc = shafftFreeBufferD(buf);
  if (rc != 0)
    TEST_FAIL("free failed");

  TEST_PASS();
}

static void test_copy_buffers_f(void) {
  TEST_BEGIN("shafftCopyTo/FromBufferF");

  const size_t count = 256;
  complexf* host_src = (complexf*)malloc(count * sizeof(complexf));
  complexf* host_dst = (complexf*)malloc(count * sizeof(complexf));
  void* device = NULL;

  for (size_t i = 0; i < count; i++) {
    host_src[i].real = (float)i;
    host_src[i].imag = (float)(i * 2);
  }

  shafftAllocBufferF(count, &device);

  int rc = shafftCopyToBufferF(device, host_src, count);
  if (rc != 0) {
    (void)shafftFreeBufferF(device);
    free(host_src);
    free(host_dst);
    TEST_FAIL("copy to failed");
  }

  rc = shafftCopyFromBufferF(host_dst, device, count);
  if (rc != 0) {
    (void)shafftFreeBufferF(device);
    free(host_src);
    free(host_dst);
    TEST_FAIL("copy from failed");
  }

  /* Verify data */
  int match = 1;
  for (size_t i = 0; i < count && match; i++) {
    if (host_src[i].real != host_dst[i].real)
      match = 0;
    if (host_src[i].imag != host_dst[i].imag)
      match = 0;
  }

  (void)shafftFreeBufferF(device);
  free(host_src);
  free(host_dst);

  if (!match)
    TEST_FAIL("data mismatch");

  TEST_PASS();
}

/*============================================================================
 * Query Tests
 *============================================================================*/

static void test_get_alloc_size(void) {
  TEST_BEGIN("shafftGetAllocSize");

  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  int dims[3] = {32, 32, 32};
  int commDims[3] = {worldSize, 1, 1};
  void* plan = NULL;

  shafftNDCreate(&plan);
  shafftNDInit(plan, 3, commDims, dims, SHAFFT_C2C, MPI_COMM_WORLD, SHAFFT_LAYOUT_REDISTRIBUTED);

  size_t localAllocSize = 0;
  int rc = shafftGetAllocSize(plan, &localAllocSize);
  if (rc != 0) {
    shafftDestroy(&plan);
    TEST_FAIL("query failed");
  }
  if (localAllocSize == 0) {
    shafftDestroy(&plan);
    TEST_FAIL("size is 0");
  }

  shafftDestroy(&plan);
  TEST_PASS();
}

static void test_get_layout(void) {
  TEST_BEGIN("shafftGetLayout");

  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  int dims[3] = {32, 32, 32};
  int commDims[3] = {worldSize, 1, 1};
  void* plan = NULL;

  shafftNDCreate(&plan);
  shafftNDInit(plan, 3, commDims, dims, SHAFFT_C2C, MPI_COMM_WORLD, SHAFFT_LAYOUT_REDISTRIBUTED);

  size_t subsize[3] = {0}, offset[3] = {0};
  int rc = shafftGetLayout(plan, subsize, offset, SHAFFT_TENSOR_LAYOUT_INITIAL);
  if (rc != 0) {
    shafftDestroy(&plan);
    TEST_FAIL("query failed");
  }

  /* Subsize should be positive */
  if (subsize[0] <= 0 || subsize[1] <= 0 || subsize[2] <= 0) {
    shafftDestroy(&plan);
    TEST_FAIL("invalid subsize");
  }

  shafftDestroy(&plan);
  TEST_PASS();
}

static void test_get_axes(void) {
  TEST_BEGIN("shafftGetAxes");

  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  int dims[3] = {32, 32, 32};
  int commDims[3] = {worldSize, 1, 1};
  void* plan = NULL;

  shafftNDCreate(&plan);
  shafftNDInit(plan, 3, commDims, dims, SHAFFT_C2C, MPI_COMM_WORLD, SHAFFT_LAYOUT_REDISTRIBUTED);

  int ca[3] = {-1, -1, -1}, da[3] = {-1, -1, -1};
  int rc = shafftGetAxes(plan, ca, da, SHAFFT_TENSOR_LAYOUT_INITIAL);
  if (rc != 0) {
    shafftDestroy(&plan);
    TEST_FAIL("query failed");
  }

  /* For nda=1 with multi-rank: da[0] should be valid axis (0) */
  if (worldSize > 1) {
    if (da[0] != 0) {
      shafftDestroy(&plan);
      TEST_FAIL("wrong distributed axis");
    }
  }

  shafftDestroy(&plan);
  TEST_PASS();
}

static void test_set_get_buffers(void) {
  TEST_BEGIN("shafftSet/GetBuffers");

  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  int dims[2] = {16, 16};
  int commDims[2] = {worldSize, 1};
  void* plan = NULL;

  shafftNDCreate(&plan);
  shafftNDInit(plan, 2, commDims, dims, SHAFFT_C2C, MPI_COMM_WORLD, SHAFFT_LAYOUT_REDISTRIBUTED);

  size_t n = 0;
  shafftGetAllocSize(plan, &n);

  void*data = NULL, *work = NULL;
  shafftAllocBufferF(n, &data);
  shafftAllocBufferF(n, &work);

  int rc = shafftSetBuffers(plan, data, work);
  shafftPlan(plan);
  if (rc != 0) {
    (void)shafftFreeBufferF(data);
    (void)shafftFreeBufferF(work);
    shafftDestroy(&plan);
    TEST_FAIL("set failed");
  }

  void*got_data = NULL, *got_work = NULL;
  rc = shafftGetBuffers(plan, &got_data, &got_work);
  if (rc != 0) {
    (void)shafftFreeBufferF(data);
    (void)shafftFreeBufferF(work);
    shafftDestroy(&plan);
    TEST_FAIL("get failed");
  }

  /* Before execute, should match what we set */
  if (got_data != data || got_work != work) {
    (void)shafftFreeBufferF(data);
    (void)shafftFreeBufferF(work);
    shafftDestroy(&plan);
    TEST_FAIL("buffers don't match");
  }

  (void)shafftFreeBufferF(data);
  (void)shafftFreeBufferF(work);
  shafftDestroy(&plan);
  TEST_PASS();
}

/*============================================================================
 * Configuration Helper Tests
 *============================================================================*/

static void test_configuration_nd(void) {
  TEST_BEGIN("shafftConfigurationND");

  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  int dims[3] = {64, 64, 32};
  int nda = 0; /* Output: number of distributed axes */
  size_t subsize[3] = {0}, offset[3] = {0};
  int commDims[3] = {0}; /* Output: communicator dimensions */
  int commSize = 0;      /* Output: product of commDims */

  int rc = shafftConfigurationND(3,
                                 dims,
                                 SHAFFT_C2C,
                                 commDims,
                                 &nda,
                                 subsize,
                                 offset,
                                 &commSize,
                                 SHAFFT_MAXIMIZE_NDA,
                                 0,
                                 MPI_COMM_WORLD);
  if (worldSize == 1) {
    /* On single rank, nda=0 (no distribution) is expected */
    if (rc != 0)
      TEST_FAIL("call failed on single rank");
    if (nda != 0)
      TEST_FAIL("expected nda=0 on single rank");
    TEST_PASS();
  }
  if (rc != 0)
    TEST_FAIL("call failed");

  /* Subsize should be positive */
  if (subsize[0] <= 0 || subsize[1] <= 0 || subsize[2] <= 0)
    TEST_FAIL("bad subsize");
  /* commSize should match worldSize */
  if (commSize != worldSize)
    TEST_FAIL("commSize mismatch");

  TEST_PASS();
}

static void test_configuration_nd_minimize(void) {
  TEST_BEGIN("shafftConfigurationND (minimize)");

  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  int dims[3] = {64, 64, 32};
  int nda = 0;
  size_t subsize[3] = {0}, offset[3] = {0};
  int commDims[3] = {0};
  int commSize = 0;

  /* Use MINIMIZE_NDA strategy - should use 1D decomposition if possible */
  int rc = shafftConfigurationND(3,
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
  if (rc != 0)
    TEST_FAIL("call failed");

  if (commSize != worldSize)
    TEST_FAIL("commSize mismatch");

  /* With minimize strategy on multi-rank, expect nda=1 if possible */
  if (worldSize > 1 && nda < 1)
    TEST_FAIL("expected nda >= 1");

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

  if (g_rank == 0) {
    printf("=== C API Coverage Tests ===\n");
    printf("SHAFFT %s (backend: %s)\n", shafftGetVersionString(), shafftGetBackendName());
    int worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    printf("MPI ranks: %d\n\n", worldSize);
  }

  /* Library info */
  test_backend_name();
  test_version();
  test_version_string();

  /* Error API */
  test_error_after_success();
  test_error_source_name();
  test_error_message();
  test_clear_error();

  /* Plan lifecycle */
  test_plan_create_destroy();
  test_plan_nd();
  test_plan_nd_2d();
  test_plan_with_init_policy();
  test_plan_invalid_output_policy();
  test_double_destroy();

  /* Buffer management */
  test_alloc_free_buffer_f();
  test_alloc_free_buffer_d();
  test_copy_buffers_f();

  /* Queries */
  test_get_alloc_size();
  test_get_layout();
  test_get_axes();
  test_set_get_buffers();

  /* Configuration helpers */
  test_configuration_nd();
  test_configuration_nd_minimize();

  if (g_rank == 0) {
    printf("\n=== C API Coverage Tests: %s ===\n", g_failed == 0 ? "PASSED" : "FAILED");
    printf("Passed: %d, Failed: %d\n", g_passed, g_failed);
  }

  MPI_Finalize();
  return g_failed > 0 ? 1 : 0;
}
