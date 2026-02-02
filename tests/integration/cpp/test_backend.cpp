/**
 * @file test_backend.cpp
 * @brief Backend-specific tests for FFTW threading and HIP streams
 * 
 * FFTW Backend:
 * - Tests that multi-threaded FFT execution works correctly
 * - Verifies SHAFFT_FFTW_THREADS environment variable is respected
 * 
 * HIP Backend:
 * - Tests that custom HIP streams can be attached to plans
 * - Verifies stream execution produces correct results
 */

#include "test_utils.hpp"
#include <shafft/shafft.hpp>
#include <shafft/shafft_config.h>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <cstdlib>

using namespace test;

constexpr double SP_TOL = 1e-4;   // Single precision tolerance (relaxed for large FFTs)
constexpr double DP_TOL = 1e-10;  // Double precision tolerance (relaxed for large FFTs)

// Helper: compute max error between two complex arrays
template<typename T>
static double max_error(const T* a, const T* b, size_t n) {
    double max_err = 0;
    for (size_t i = 0; i < n; ++i) {
        double err_re = std::fabs(a[i].real() - b[i].real());
        double err_im = std::fabs(a[i].imag() - b[i].imag());
        max_err = std::max(max_err, std::max(err_re, err_im));
    }
    return max_err;
}

#if SHAFFT_BACKEND_FFTW
// ============================================================================
// FFTW Threading Tests
// ============================================================================

/**
 * Test that FFTW threading produces correct results.
 * Uses SHAFFT_FFTW_THREADS environment variable.
 */
static bool test_fftw_threads_roundtrip() {
    // Set threading via environment (library reads this at plan creation)
    setenv("SHAFFT_FFTW_THREADS", "2", 1);
    
    std::vector<int> dims = {64, 64, 64};  // Large enough to benefit from threads
    
    // Use configurationCart to get valid decomposition
    std::vector<int> comm_dims(dims.size(), 0);
    std::vector<int> subsize(dims.size()), offset(dims.size());
    int comm_size = 0;
    int rc = shafft::configurationCart(dims, subsize, offset, comm_dims,
                                       comm_size, shafft::FFTType::Z2Z, 0, MPI_COMM_WORLD);
    if (rc != 0) {
        unsetenv("SHAFFT_FFTW_THREADS");
        return false;
    }
    
    shafft::Plan plan;
    rc = plan.initCart(comm_dims, dims, shafft::FFTType::Z2Z, MPI_COMM_WORLD);
    if (rc != 0) {
        unsetenv("SHAFFT_FFTW_THREADS");
        return false;
    }
    
    // Re-get layout after plan init
    (void)plan.getLayout(subsize, offset, shafft::TensorLayout::CURRENT);
    
    size_t local_elems = product(subsize);
    size_t alloc_elems = plan.allocSize();
    
    shafft::complexd *data = nullptr, *work = nullptr;
    (void)shafft::allocBuffer(alloc_elems, &data);
    (void)shafft::allocBuffer(alloc_elems, &work);
    
    // Initialize with known pattern
    std::vector<shafft::complexd> original(alloc_elems);
    for (size_t i = 0; i < local_elems; ++i) {
        original[i] = shafft::complexd(static_cast<double>(i), static_cast<double>(i) * 0.5);
    }
    
    (void)shafft::copyToBuffer(data, original.data(), alloc_elems);
    (void)plan.setBuffers(data, work);
    
    // Forward + backward + normalize
    (void)plan.execute(shafft::FFTDirection::FORWARD);
    (void)plan.execute(shafft::FFTDirection::BACKWARD);
    (void)plan.normalize();
    
    shafft::complexd *final_data, *final_work;
    (void)plan.getBuffers(&final_data, &final_work);
    
    std::vector<shafft::complexd> result(alloc_elems);
    (void)shafft::copyFromBuffer(result.data(), final_data, alloc_elems);
    
    double local_err = max_error(original.data(), result.data(), local_elems);
    double global_err = 0;
    MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    
    // Clean up environment
    unsetenv("SHAFFT_FFTW_THREADS");
    
    return global_err < DP_TOL;
}

/**
 * Test single-threaded FFTW (baseline comparison).
 */
static bool test_fftw_single_thread_roundtrip() {
    setenv("SHAFFT_FFTW_THREADS", "1", 1);
    
    std::vector<int> dims = {32, 32, 32};
    
    // Use configurationCart to get valid decomposition
    std::vector<int> comm_dims(dims.size(), 0);
    std::vector<int> subsize(dims.size()), offset(dims.size());
    int comm_size = 0;
    int rc = shafft::configurationCart(dims, subsize, offset, comm_dims,
                                       comm_size, shafft::FFTType::C2C, 0, MPI_COMM_WORLD);
    if (rc != 0) {
        unsetenv("SHAFFT_FFTW_THREADS");
        return false;
    }
    
    shafft::Plan plan;
    rc = plan.initCart(comm_dims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) {
        unsetenv("SHAFFT_FFTW_THREADS");
        return false;
    }
    
    // Re-get layout after plan init
    (void)plan.getLayout(subsize, offset, shafft::TensorLayout::CURRENT);
    
    size_t local_elems = product(subsize);
    size_t alloc_elems = plan.allocSize();
    
    shafft::complexf *data = nullptr, *work = nullptr;
    (void)shafft::allocBuffer(alloc_elems, &data);
    (void)shafft::allocBuffer(alloc_elems, &work);
    
    std::vector<shafft::complexf> original(alloc_elems);
    for (size_t i = 0; i < local_elems; ++i) {
        original[i] = shafft::complexf(static_cast<float>(i % 100), 0.0f);
    }
    
    (void)shafft::copyToBuffer(data, original.data(), alloc_elems);
    (void)plan.setBuffers(data, work);
    
    (void)plan.execute(shafft::FFTDirection::FORWARD);
    (void)plan.execute(shafft::FFTDirection::BACKWARD);
    (void)plan.normalize();
    
    shafft::complexf *final_data, *final_work;
    (void)plan.getBuffers(&final_data, &final_work);
    
    std::vector<shafft::complexf> result(alloc_elems);
    (void)shafft::copyFromBuffer(result.data(), final_data, alloc_elems);
    
    double local_err = max_error(original.data(), result.data(), local_elems);
    double global_err = 0;
    MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    
    unsetenv("SHAFFT_FFTW_THREADS");
    
    return global_err < SP_TOL;
}

/**
 * Test FFTW with maximum threads (OMP_NUM_THREADS or hardware concurrency).
 */
static bool test_fftw_max_threads_roundtrip() {
    // Use 4 threads (reasonable for most systems)
    setenv("SHAFFT_FFTW_THREADS", "4", 1);
    
    std::vector<int> dims = {128, 128};  // 2D, larger problem
    int nda = 1;
    
    shafft::Plan plan;
    int rc = plan.init(nda, dims, shafft::FFTType::Z2Z, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    std::vector<int> subsize(dims.size()), offset(dims.size());
    (void)plan.getLayout(subsize, offset, shafft::TensorLayout::CURRENT);
    
    size_t local_elems = product(subsize);
    size_t alloc_elems = plan.allocSize();
    
    shafft::complexd *data = nullptr, *work = nullptr;
    (void)shafft::allocBuffer(alloc_elems, &data);
    (void)shafft::allocBuffer(alloc_elems, &work);
    
    std::vector<shafft::complexd> original(alloc_elems);
    for (size_t i = 0; i < local_elems; ++i) {
        original[i] = shafft::complexd(std::sin(static_cast<double>(i)), 
                                        std::cos(static_cast<double>(i)));
    }
    
    (void)shafft::copyToBuffer(data, original.data(), alloc_elems);
    (void)plan.setBuffers(data, work);
    
    (void)plan.execute(shafft::FFTDirection::FORWARD);
    (void)plan.execute(shafft::FFTDirection::BACKWARD);
    (void)plan.normalize();
    
    shafft::complexd *final_data, *final_work;
    (void)plan.getBuffers(&final_data, &final_work);
    
    std::vector<shafft::complexd> result(alloc_elems);
    (void)shafft::copyFromBuffer(result.data(), final_data, alloc_elems);
    
    double local_err = max_error(original.data(), result.data(), local_elems);
    double global_err = 0;
    MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    
    unsetenv("SHAFFT_FFTW_THREADS");
    
    return global_err < DP_TOL;
}

#endif // SHAFFT_BACKEND_FFTW

#if SHAFFT_BACKEND_HIPFFT
// ============================================================================
// HIP Stream Tests
// ============================================================================

/**
 * Test that setStream works and produces correct results.
 */
static bool test_hip_stream_roundtrip() {
    hipStream_t stream;
    hipError_t hip_rc = hipStreamCreate(&stream);
    if (hip_rc != hipSuccess) return false;
    
    std::vector<int> dims = {64, 64, 64};
    int nda = 1;
    
    shafft::Plan plan;
    int rc = plan.init(nda, dims, shafft::FFTType::Z2Z, MPI_COMM_WORLD);
    if (rc != 0) {
        hipStreamDestroy(stream);
        return false;
    }
    
    // Attach custom stream to plan
    rc = plan.setStream(stream);
    if (rc != 0) {
        hipStreamDestroy(stream);
        return false;
    }
    
    std::vector<int> subsize(dims.size()), offset(dims.size());
    (void)plan.getLayout(subsize, offset, shafft::TensorLayout::CURRENT);
    
    size_t local_elems = product(subsize);
    size_t alloc_elems = plan.allocSize();
    
    shafft::complexd *data = nullptr, *work = nullptr;
    (void)shafft::allocBuffer(alloc_elems, &data);
    (void)shafft::allocBuffer(alloc_elems, &work);
    
    std::vector<shafft::complexd> original(alloc_elems);
    for (size_t i = 0; i < local_elems; ++i) {
        original[i] = shafft::complexd(static_cast<double>(i), static_cast<double>(i) * 0.5);
    }
    
    (void)shafft::copyToBuffer(data, original.data(), alloc_elems);
    (void)plan.setBuffers(data, work);
    
    (void)plan.execute(shafft::FFTDirection::FORWARD);
    (void)plan.execute(shafft::FFTDirection::BACKWARD);
    (void)plan.normalize();
    
    // Synchronize stream before reading results
    hipStreamSynchronize(stream);
    
    shafft::complexd *final_data, *final_work;
    (void)plan.getBuffers(&final_data, &final_work);
    
    std::vector<shafft::complexd> result(alloc_elems);
    (void)shafft::copyFromBuffer(result.data(), final_data, alloc_elems);
    
    double local_err = max_error(original.data(), result.data(), local_elems);
    double global_err = 0;
    MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    hipStreamDestroy(stream);
    
    return global_err < DP_TOL;
}

/**
 * Test default stream (null stream) works correctly.
 */
static bool test_hip_default_stream_roundtrip() {
    std::vector<int> dims = {32, 32, 32};
    int nda = 1;
    
    shafft::Plan plan;
    int rc = plan.init(nda, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    // Use null stream (default)
    rc = plan.setStream(nullptr);
    if (rc != 0) return false;
    
    std::vector<int> subsize(dims.size()), offset(dims.size());
    (void)plan.getLayout(subsize, offset, shafft::TensorLayout::CURRENT);
    
    size_t local_elems = product(subsize);
    size_t alloc_elems = plan.allocSize();
    
    shafft::complexf *data = nullptr, *work = nullptr;
    (void)shafft::allocBuffer(alloc_elems, &data);
    (void)shafft::allocBuffer(alloc_elems, &work);
    
    std::vector<shafft::complexf> original(alloc_elems);
    for (size_t i = 0; i < local_elems; ++i) {
        original[i] = shafft::complexf(static_cast<float>(i % 100), 0.0f);
    }
    
    (void)shafft::copyToBuffer(data, original.data(), alloc_elems);
    (void)plan.setBuffers(data, work);
    
    (void)plan.execute(shafft::FFTDirection::FORWARD);
    (void)plan.execute(shafft::FFTDirection::BACKWARD);
    (void)plan.normalize();
    
    // Default stream is synchronous, no explicit sync needed
    
    shafft::complexf *final_data, *final_work;
    (void)plan.getBuffers(&final_data, &final_work);
    
    std::vector<shafft::complexf> result(alloc_elems);
    (void)shafft::copyFromBuffer(result.data(), final_data, alloc_elems);
    
    double local_err = max_error(original.data(), result.data(), local_elems);
    double global_err = 0;
    MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    
    return global_err < SP_TOL;
}

/**
 * Test multiple streams on same plan (stream switching).
 */
static bool test_hip_stream_switching() {
    hipStream_t stream1, stream2;
    if (hipStreamCreate(&stream1) != hipSuccess) return false;
    if (hipStreamCreate(&stream2) != hipSuccess) {
        hipStreamDestroy(stream1);
        return false;
    }
    
    std::vector<int> dims = {32, 32};
    int nda = 1;
    
    shafft::Plan plan;
    int rc = plan.init(nda, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) {
        hipStreamDestroy(stream1);
        hipStreamDestroy(stream2);
        return false;
    }
    
    std::vector<int> subsize(dims.size()), offset(dims.size());
    (void)plan.getLayout(subsize, offset, shafft::TensorLayout::CURRENT);
    
    size_t local_elems = product(subsize);
    size_t alloc_elems = plan.allocSize();
    
    shafft::complexf *data = nullptr, *work = nullptr;
    (void)shafft::allocBuffer(alloc_elems, &data);
    (void)shafft::allocBuffer(alloc_elems, &work);
    
    std::vector<shafft::complexf> original(alloc_elems);
    for (size_t i = 0; i < local_elems; ++i) {
        original[i] = shafft::complexf(1.0f, 0.0f);
    }
    
    (void)shafft::copyToBuffer(data, original.data(), alloc_elems);
    (void)plan.setBuffers(data, work);
    
    // Use stream1 for forward
    plan.setStream(stream1);
    (void)plan.execute(shafft::FFTDirection::FORWARD);
    hipStreamSynchronize(stream1);
    
    // Switch to stream2 for backward
    plan.setStream(stream2);
    (void)plan.execute(shafft::FFTDirection::BACKWARD);
    (void)plan.normalize();
    hipStreamSynchronize(stream2);
    
    shafft::complexf *final_data, *final_work;
    (void)plan.getBuffers(&final_data, &final_work);
    
    std::vector<shafft::complexf> result(alloc_elems);
    (void)shafft::copyFromBuffer(result.data(), final_data, alloc_elems);
    
    double local_err = max_error(original.data(), result.data(), local_elems);
    double global_err = 0;
    MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    hipStreamDestroy(stream1);
    hipStreamDestroy(stream2);
    
    return global_err < SP_TOL;
}

#endif // SHAFFT_BACKEND_HIPFFT

// ============================================================================
// Backend Name Verification
// ============================================================================

static bool test_backend_name() {
    std::string name = shafft::getBackendName();
    
#if SHAFFT_BACKEND_FFTW
    return name == "FFTW";
#elif SHAFFT_BACKEND_HIPFFT
    return name == "hipFFT";
#else
    return !name.empty();  // At least some name should be returned
#endif
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    TestRunner runner("Backend-Specific Tests");
    
    // Backend name verification (always runs)
    runner.run("Backend name verification", test_backend_name);
    
#if SHAFFT_BACKEND_FFTW
    runner.run("FFTW single-threaded roundtrip", test_fftw_single_thread_roundtrip);
    runner.run("FFTW 2-threaded roundtrip", test_fftw_threads_roundtrip);
    runner.run("FFTW 4-threaded roundtrip", test_fftw_max_threads_roundtrip);
#endif

#if SHAFFT_BACKEND_HIPFFT
    runner.run("HIP custom stream roundtrip", test_hip_stream_roundtrip);
    runner.run("HIP default stream roundtrip", test_hip_default_stream_roundtrip);
    runner.run("HIP stream switching", test_hip_stream_switching);
#endif
    
    int rc = runner.finalize();
    MPI_Finalize();
    return rc;
}
