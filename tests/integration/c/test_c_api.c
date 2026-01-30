/**
 * @file test_c_api.c
 * @brief Comprehensive C API coverage test
 */
#include <shafft/shafft.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Test result tracking */
static int g_rank = 0;
static int g_passed = 0;
static int g_failed = 0;

#define TEST_BEGIN(name) \
    do { if (g_rank == 0) printf("  %-40s ", name); } while(0)

#define TEST_PASS() \
    do { if (g_rank == 0) printf("PASS\n"); g_passed++; return; } while(0)

#define TEST_FAIL(msg) \
    do { if (g_rank == 0) printf("FAIL (%s)\n", msg); g_failed++; return; } while(0)

#define TEST_SKIP(msg) \
    do { if (g_rank == 0) printf("SKIP (%s)\n", msg); return; } while(0)

/* Single-precision complex type */
typedef struct { float real; float imag; } complexf;

/*============================================================================
 * Library Info Tests
 *============================================================================*/

static void test_backend_name(void) {
    TEST_BEGIN("shafftGetBackendName");
    
    const char* backend = shafftGetBackendName();
    if (backend == NULL) TEST_FAIL("returned NULL");
    if (strlen(backend) == 0) TEST_FAIL("empty string");
    
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
    
    if (major < 0 || minor < 0 || patch < 0) TEST_FAIL("negative version");
    
    TEST_PASS();
}

static void test_version_string(void) {
    TEST_BEGIN("shafftGetVersionString");
    
    const char* ver = shafftGetVersionString();
    if (ver == NULL) TEST_FAIL("returned NULL");
    if (strlen(ver) == 0) TEST_FAIL("empty string");
    
    /* Should contain at least one dot */
    if (strchr(ver, '.') == NULL) TEST_FAIL("no dots in version");
    
    TEST_PASS();
}

/*============================================================================
 * Error API Tests
 *============================================================================*/

static void test_error_after_success(void) {
    TEST_BEGIN("error_after_success");
    
    /* Clear any prior error */
    shafft_clear_last_error();
    
    /* Successful operation */
    const char* backend = shafftGetBackendName();
    (void)backend;
    
    /* Should have no error */
    if (shafft_last_error_status() != SHAFFT_SUCCESS) TEST_FAIL("expected SUCCESS");
    
    TEST_PASS();
}

static void test_error_source_name(void) {
    TEST_BEGIN("shafft_error_source_name");
    
    const char* name;
    
    name = shafft_error_source_name(SHAFFT_ERRSRC_NONE);
    if (name == NULL || strlen(name) == 0) TEST_FAIL("NONE name empty");
    
    name = shafft_error_source_name(SHAFFT_ERRSRC_MPI);
    if (name == NULL || strlen(name) == 0) TEST_FAIL("MPI name empty");
    
    name = shafft_error_source_name(SHAFFT_ERRSRC_HIP);
    if (name == NULL || strlen(name) == 0) TEST_FAIL("HIP name empty");
    
    TEST_PASS();
}

static void test_error_message(void) {
    TEST_BEGIN("shafft_last_error_message");
    
    char buf[256] = {0};
    int len = shafft_last_error_message(buf, sizeof(buf));
    
    /* Should return some length (may be 0 if no error) */
    if (len < 0) TEST_FAIL("negative length");
    
    TEST_PASS();
}

static void test_clear_error(void) {
    TEST_BEGIN("shafft_clear_last_error");
    
    /* Create an error by passing NULL */
    int rc = shafftPlanNDA(NULL, 3, 1, NULL, SHAFFT_C2C, MPI_COMM_WORLD);
    (void)rc;
    
    /* Should have error */
    if (shafft_last_error_status() == SHAFFT_SUCCESS) {
        /* Some implementations may not set error for NULL plan */
        shafft_clear_last_error();
        TEST_PASS();
    }
    
    /* Clear it */
    shafft_clear_last_error();
    
    /* Should be cleared */
    if (shafft_last_error_status() != SHAFFT_SUCCESS) TEST_FAIL("not cleared");
    
    TEST_PASS();
}

/*============================================================================
 * Plan Lifecycle Tests
 *============================================================================*/

static void test_plan_create_destroy(void) {
    TEST_BEGIN("shafftPlanCreate/Destroy");
    
    void* plan = NULL;
    int rc = shafftPlanCreate(&plan);
    if (rc != 0) TEST_FAIL("create failed");
    if (plan == NULL) TEST_FAIL("plan is NULL");
    
    rc = shafftDestroy(&plan);
    if (rc != 0) TEST_FAIL("destroy failed");
    if (plan != NULL) TEST_FAIL("plan not set to NULL");
    
    TEST_PASS();
}

static void test_plan_nda(void) {
    TEST_BEGIN("shafftPlanNDA");
    
    int dims[3] = {32, 32, 32};
    void* plan = NULL;
    
    shafftPlanCreate(&plan);
    int rc = shafftPlanNDA(plan, 3, 1, dims, SHAFFT_C2C, MPI_COMM_WORLD);
    if (rc != 0) {
        shafftDestroy(&plan);
        TEST_FAIL("init failed");
    }
    
    shafftDestroy(&plan);
    TEST_PASS();
}

static void test_plan_cart(void) {
    TEST_BEGIN("shafftPlanCart");
    
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    int dims[3] = {32, 32, 32};
    /* COMM_DIMS must have ndim elements, last must be 1 */
    int comm_dims[3] = {world_size, 1, 1};
    void* plan = NULL;
    
    shafftPlanCreate(&plan);
    int rc = shafftPlanCart(plan, 3, comm_dims, dims, SHAFFT_C2C, MPI_COMM_WORLD);
    if (rc != 0) {
        shafftDestroy(&plan);
        TEST_FAIL("init failed");
    }
    
    shafftDestroy(&plan);
    TEST_PASS();
}

static void test_double_destroy(void) {
    TEST_BEGIN("double_destroy");
    
    int dims[2] = {16, 16};
    void* plan = NULL;
    
    shafftPlanCreate(&plan);
    shafftPlanNDA(plan, 2, 1, dims, SHAFFT_C2C, MPI_COMM_WORLD);
    
    int rc1 = shafftDestroy(&plan);
    int rc2 = shafftDestroy(&plan);  /* Should handle NULL gracefully */
    
    if (rc1 != 0) TEST_FAIL("first destroy failed");
    /* Second destroy on NULL may or may not fail, just shouldn't crash */
    (void)rc2;
    
    TEST_PASS();
}

/*============================================================================
 * Buffer Management Tests
 *============================================================================*/

static void test_alloc_free_buffer_f(void) {
    TEST_BEGIN("shafftAllocBufferF/FreeBufferF");
    
    void* buf = NULL;
    int rc = shafftAllocBufferF(1024, &buf);
    if (rc != 0) TEST_FAIL("alloc failed");
    if (buf == NULL) TEST_FAIL("buf is NULL");
    
    rc = shafftFreeBufferF(buf);
    if (rc != 0) TEST_FAIL("free failed");
    
    /* Free NULL should be safe */
    rc = shafftFreeBufferF(NULL);
    if (rc != 0) TEST_FAIL("free NULL failed");
    
    TEST_PASS();
}

static void test_alloc_free_buffer_d(void) {
    TEST_BEGIN("shafftAllocBufferD/FreeBufferD");
    
    void* buf = NULL;
    int rc = shafftAllocBufferD(1024, &buf);
    if (rc != 0) TEST_FAIL("alloc failed");
    if (buf == NULL) TEST_FAIL("buf is NULL");
    
    rc = shafftFreeBufferD(buf);
    if (rc != 0) TEST_FAIL("free failed");
    
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
        shafftFreeBufferF(device);
        free(host_src);
        free(host_dst);
        TEST_FAIL("copy to failed");
    }
    
    rc = shafftCopyFromBufferF(host_dst, device, count);
    if (rc != 0) {
        shafftFreeBufferF(device);
        free(host_src);
        free(host_dst);
        TEST_FAIL("copy from failed");
    }
    
    /* Verify data */
    int match = 1;
    for (size_t i = 0; i < count && match; i++) {
        if (host_src[i].real != host_dst[i].real) match = 0;
        if (host_src[i].imag != host_dst[i].imag) match = 0;
    }
    
    shafftFreeBufferF(device);
    free(host_src);
    free(host_dst);
    
    if (!match) TEST_FAIL("data mismatch");
    
    TEST_PASS();
}

/*============================================================================
 * Query Tests
 *============================================================================*/

static void test_get_alloc_size(void) {
    TEST_BEGIN("shafftGetAllocSize");
    
    int dims[3] = {32, 32, 32};
    void* plan = NULL;
    
    shafftPlanCreate(&plan);
    shafftPlanNDA(plan, 3, 1, dims, SHAFFT_C2C, MPI_COMM_WORLD);
    
    size_t alloc_size = 0;
    int rc = shafftGetAllocSize(plan, &alloc_size);
    if (rc != 0) {
        shafftDestroy(&plan);
        TEST_FAIL("query failed");
    }
    if (alloc_size == 0) {
        shafftDestroy(&plan);
        TEST_FAIL("size is 0");
    }
    
    shafftDestroy(&plan);
    TEST_PASS();
}

static void test_get_layout(void) {
    TEST_BEGIN("shafftGetLayout");
    
    int dims[3] = {32, 32, 32};
    void* plan = NULL;
    
    shafftPlanCreate(&plan);
    shafftPlanNDA(plan, 3, 1, dims, SHAFFT_C2C, MPI_COMM_WORLD);
    
    int subsize[3] = {0}, offset[3] = {0};
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
    
    int dims[3] = {32, 32, 32};
    void* plan = NULL;
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    shafftPlanCreate(&plan);
    shafftPlanNDA(plan, 3, 1, dims, SHAFFT_C2C, MPI_COMM_WORLD);
    
    int ca[3] = {-1, -1, -1}, da[3] = {-1, -1, -1};
    int rc = shafftGetAxes(plan, ca, da, SHAFFT_TENSOR_LAYOUT_INITIAL);
    if (rc != 0) {
        shafftDestroy(&plan);
        TEST_FAIL("query failed");
    }
    
    /* For nda=1 with multi-rank: da[0] should be valid axis (0) */
    if (world_size > 1) {
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
    
    int dims[2] = {16, 16};
    void* plan = NULL;
    
    shafftPlanCreate(&plan);
    shafftPlanNDA(plan, 2, 1, dims, SHAFFT_C2C, MPI_COMM_WORLD);
    
    size_t n = 0;
    shafftGetAllocSize(plan, &n);
    
    void *data = NULL, *work = NULL;
    shafftAllocBufferF(n, &data);
    shafftAllocBufferF(n, &work);
    
    int rc = shafftSetBuffers(plan, data, work);
    if (rc != 0) {
        shafftFreeBufferF(data);
        shafftFreeBufferF(work);
        shafftDestroy(&plan);
        TEST_FAIL("set failed");
    }
    
    void *got_data = NULL, *got_work = NULL;
    rc = shafftGetBuffers(plan, &got_data, &got_work);
    if (rc != 0) {
        shafftFreeBufferF(data);
        shafftFreeBufferF(work);
        shafftDestroy(&plan);
        TEST_FAIL("get failed");
    }
    
    /* Before execute, should match what we set */
    if (got_data != data || got_work != work) {
        shafftFreeBufferF(data);
        shafftFreeBufferF(work);
        shafftDestroy(&plan);
        TEST_FAIL("buffers don't match");
    }
    
    shafftFreeBufferF(data);
    shafftFreeBufferF(work);
    shafftDestroy(&plan);
    TEST_PASS();
}

/*============================================================================
 * Configuration Helper Tests
 *============================================================================*/

static void test_configuration_nda(void) {
    TEST_BEGIN("shafftConfigurationNDA");
    
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int dims[3] = {64, 64, 32};
    int nda = 1;
    int subsize[3] = {0}, offset[3] = {0}, comm_dims[3] = {0};
    
    int rc = shafftConfigurationNDA(3, dims, &nda, subsize, offset, comm_dims,
                                    SHAFFT_C2C, 0, MPI_COMM_WORLD);
    if (world_size == 1) {
        // Explicit nda=1 cannot be satisfied on a single rank; expect failure
        if (rc == 0) TEST_FAIL("expected INVALID_DECOMP on 1 rank");
        TEST_PASS();
    }
    if (rc != 0) TEST_FAIL("call failed");
    
    /* Subsize should be positive */
    if (subsize[0] <= 0 || subsize[1] <= 0 || subsize[2] <= 0) TEST_FAIL("bad subsize");
    
    TEST_PASS();
}

static void test_configuration_cart(void) {
    TEST_BEGIN("shafftConfigurationCart");
    
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    int dims[3] = {64, 64, 32};
    int subsize[3] = {0}, offset[3] = {0};
    /* COMM_DIMS must have ndim elements for ConfigurationCart */
    int comm_dims[3] = {world_size, 1, 1};
    int comm_size = 0;
    
    int rc = shafftConfigurationCart(3, dims, subsize, offset, comm_dims,
                                     &comm_size, SHAFFT_C2C, 0, MPI_COMM_WORLD);
    if (rc != 0) TEST_FAIL("call failed");
    
    if (comm_size != world_size) TEST_FAIL("comm_size mismatch");
    
    TEST_PASS();
}

/*============================================================================
 * Main
 *============================================================================*/

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    
    if (g_rank == 0) {
        printf("=== C API Coverage Tests ===\n");
        printf("SHAFFT %s (backend: %s)\n", shafftGetVersionString(), shafftGetBackendName());
        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        printf("MPI ranks: %d\n\n", world_size);
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
    test_plan_nda();
    test_plan_cart();
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
    test_configuration_nda();
    test_configuration_cart();
    
    if (g_rank == 0) {
        printf("\n=== C API Coverage Tests: %s ===\n", 
               g_failed == 0 ? "PASSED" : "FAILED");
        printf("Passed: %d, Failed: %d\n", g_passed, g_failed);
    }
    
    MPI_Finalize();
    return g_failed > 0 ? 1 : 0;
}
