/**
 * @file test_c_roundtrip.c
 * @brief Test that the C API can perform a basic forward/backward/normalize roundtrip
 */
#include <shafft/shafft.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Single-precision complex type (matches shafft's layout) */
typedef struct { float real; float imag; } complexf;

/* Tolerance for floating point comparison */
#define TOLERANCE 1e-4f

#define FAIL_WITH_ERR(label, rc) \
    do { \
        char buf[256] = {0}; \
        shafft_last_error_message(buf, sizeof(buf)); \
        fprintf(stderr, "%s failed rc=%d err=\"%s\"\n", label, rc, buf); \
        return 1; \
    } while (0)

static int test_roundtrip_basic(void) {
    int dims[3] = {32, 32, 32};
    const int ndim = 3;
    const int nda = 1;
    int rc;
    
    /* Create plan */
    void* plan = NULL;
    rc = shafftPlanCreate(&plan);
    if (rc != 0 || plan == NULL) FAIL_WITH_ERR("shafftPlanCreate", rc);
    
    /* Initialize plan */
    rc = shafftPlanNDA(plan, ndim, nda, dims, SHAFFT_C2C, MPI_COMM_WORLD);
    if (rc != 0) { shafftDestroy(&plan); FAIL_WITH_ERR("shafftPlanNDA", rc); }
    
    /* Get allocation size */
    size_t alloc_size = 0;
    rc = shafftGetAllocSize(plan, &alloc_size);
    if (rc != 0 || alloc_size == 0) { shafftDestroy(&plan); FAIL_WITH_ERR("shafftGetAllocSize", rc); }
    
    /* Allocate device buffers */
    void *data = NULL, *work = NULL;
    rc = shafftAllocBufferF(alloc_size, &data);
    if (rc != 0) { shafftDestroy(&plan); FAIL_WITH_ERR("shafftAllocBufferF(data)", rc); }
    rc = shafftAllocBufferF(alloc_size, &work);
    if (rc != 0) { shafftFreeBufferF(data); shafftDestroy(&plan); FAIL_WITH_ERR("shafftAllocBufferF(work)", rc); }
    
    /* Create host data with a known pattern */
    complexf* host_orig = (complexf*)malloc(alloc_size * sizeof(complexf));
    complexf* host_result = (complexf*)malloc(alloc_size * sizeof(complexf));
    if (!host_orig || !host_result) {
        fprintf(stderr, "Host allocation failed\n");
        shafftFreeBufferF(data);
        shafftFreeBufferF(work);
        shafftDestroy(&plan);
        free(host_orig);
        free(host_result);
        return 1;
    }
    
    /* Initialize with a pattern */
    for (size_t i = 0; i < alloc_size; i++) {
        host_orig[i].real = (float)(i % 100) / 100.0f;
        host_orig[i].imag = (float)((i + 50) % 100) / 100.0f;
    }
    
    /* Copy to device */
    rc = shafftCopyToBufferF(data, host_orig, alloc_size);
    if (rc != 0) { shafftFreeBufferF(data); shafftFreeBufferF(work); shafftDestroy(&plan); free(host_orig); free(host_result); FAIL_WITH_ERR("shafftCopyToBufferF", rc); }
    
    /* Set buffers */
    rc = shafftSetBuffers(plan, data, work);
    if (rc != 0) { shafftFreeBufferF(data); shafftFreeBufferF(work); shafftDestroy(&plan); free(host_orig); free(host_result); FAIL_WITH_ERR("shafftSetBuffers", rc); }
    
    /* Execute forward transform */
    rc = shafftExecute(plan, SHAFFT_FORWARD);
    if (rc != 0) { shafftFreeBufferF(data); shafftFreeBufferF(work); shafftDestroy(&plan); free(host_orig); free(host_result); FAIL_WITH_ERR("shafftExecute(FORWARD)", rc); }
    
    /* Execute backward transform */
    rc = shafftExecute(plan, SHAFFT_BACKWARD);
    if (rc != 0) { shafftFreeBufferF(data); shafftFreeBufferF(work); shafftDestroy(&plan); free(host_orig); free(host_result); FAIL_WITH_ERR("shafftExecute(BACKWARD)", rc); }
    
    /* Normalize */
    rc = shafftNormalize(plan);
    if (rc != 0) { shafftFreeBufferF(data); shafftFreeBufferF(work); shafftDestroy(&plan); free(host_orig); free(host_result); FAIL_WITH_ERR("shafftNormalize", rc); }
    
    /* Get the result buffer (may have swapped) */
    void *result_data = NULL, *result_work = NULL;
    rc = shafftGetBuffers(plan, &result_data, &result_work);
    if (rc != 0) { shafftFreeBufferF(data); shafftFreeBufferF(work); shafftDestroy(&plan); free(host_orig); free(host_result); FAIL_WITH_ERR("shafftGetBuffers", rc); }
    
    /* Copy result back to host */
    rc = shafftCopyFromBufferF(host_result, result_data, alloc_size);
    if (rc != 0) { shafftFreeBufferF(data); shafftFreeBufferF(work); shafftDestroy(&plan); free(host_orig); free(host_result); FAIL_WITH_ERR("shafftCopyFromBufferF", rc); }
    
    /* Compare with original */
    float max_err = 0.0f;
    for (size_t i = 0; i < alloc_size; i++) {
        float err_real = fabsf(host_result[i].real - host_orig[i].real);
        float err_imag = fabsf(host_result[i].imag - host_orig[i].imag);
        if (err_real > max_err) max_err = err_real;
        if (err_imag > max_err) max_err = err_imag;
    }
    
    int pass = (max_err < TOLERANCE);
    
    /* Cleanup */
    shafftFreeBufferF(data);
    shafftFreeBufferF(work);
    shafftDestroy(&plan);
    free(host_orig);
    free(host_result);
    
    if (!pass) {
        fprintf(stderr, "Roundtrip error too large: %e (tolerance: %e)\n", max_err, TOLERANCE);
        return 1;
    }
    
    return 0;
}

static int test_roundtrip_double(void) {
    int dims[2] = {64, 64};
    const int ndim = 2;
    const int nda = 1;
    int rc;
    
    /* Double-precision complex type */
    typedef struct { double real; double imag; } complexd;
    
    /* Create plan */
    void* plan = NULL;
    rc = shafftPlanCreate(&plan);
    if (rc != 0) return 1;
    
    rc = shafftPlanNDA(plan, ndim, nda, dims, SHAFFT_Z2Z, MPI_COMM_WORLD);
    if (rc != 0) {
        shafftDestroy(&plan);
        return 1;
    }
    
    size_t alloc_size = 0;
    rc = shafftGetAllocSize(plan, &alloc_size);
    if (rc != 0) {
        shafftDestroy(&plan);
        return 1;
    }
    
    void *data = NULL, *work = NULL;
    rc = shafftAllocBufferD(alloc_size, &data);
    if (rc != 0) {
        shafftDestroy(&plan);
        return 1;
    }
    rc = shafftAllocBufferD(alloc_size, &work);
    if (rc != 0) {
        shafftFreeBufferD(data);
        shafftDestroy(&plan);
        return 1;
    }
    
    complexd* host_orig = (complexd*)malloc(alloc_size * sizeof(complexd));
    complexd* host_result = (complexd*)malloc(alloc_size * sizeof(complexd));
    
    for (size_t i = 0; i < alloc_size; i++) {
        host_orig[i].real = (double)(i % 100) / 100.0;
        host_orig[i].imag = (double)((i + 50) % 100) / 100.0;
    }
    
    shafftCopyToBufferD(data, host_orig, alloc_size);
    shafftSetBuffers(plan, data, work);
    shafftExecute(plan, SHAFFT_FORWARD);
    shafftExecute(plan, SHAFFT_BACKWARD);
    shafftNormalize(plan);
    
    void *result_data = NULL, *result_work = NULL;
    shafftGetBuffers(plan, &result_data, &result_work);
    shafftCopyFromBufferD(host_result, result_data, alloc_size);
    
    double max_err = 0.0;
    for (size_t i = 0; i < alloc_size; i++) {
        double err_real = fabs(host_result[i].real - host_orig[i].real);
        double err_imag = fabs(host_result[i].imag - host_orig[i].imag);
        if (err_real > max_err) max_err = err_real;
        if (err_imag > max_err) max_err = err_imag;
    }
    
    int pass = (max_err < 1e-10);
    
    shafftFreeBufferD(data);
    shafftFreeBufferD(work);
    shafftDestroy(&plan);
    free(host_orig);
    free(host_result);
    
    if (!pass) {
        fprintf(stderr, "Double precision roundtrip error: %e\n", max_err);
        return 1;
    }
    
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int failed = 0;
    int total = 0;
    
    if (rank == 0) {
        printf("=== C API Roundtrip Tests ===\n");
        printf("SHAFFT %s (backend: %s)\n", shafftGetVersionString(), shafftGetBackendName());
        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        printf("MPI ranks: %d\n\n", world_size);
    }
    
    /* Test 1: Single precision roundtrip */
    total++;
    if (rank == 0) printf("  roundtrip_basic                          ");
    if (test_roundtrip_basic() == 0) {
        if (rank == 0) printf("PASS\n");
    } else {
        if (rank == 0) printf("FAIL\n");
        failed++;
    }
    
    /* Test 2: Double precision roundtrip */
    total++;
    if (rank == 0) printf("  roundtrip_double                         ");
    if (test_roundtrip_double() == 0) {
        if (rank == 0) printf("PASS\n");
    } else {
        if (rank == 0) printf("FAIL\n");
        failed++;
    }
    
    if (rank == 0) {
        printf("\n=== C API Roundtrip Tests: %s ===\n", failed == 0 ? "PASSED" : "FAILED");
    }
    
    MPI_Finalize();
    return failed > 0 ? 1 : 0;
}
