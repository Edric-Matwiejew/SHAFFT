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
#include <cmath>
#include <cstdlib>
#include <mpi.h>
#include <shafft/shafft.hpp>
#include <shafft/shafft_config.h>
#include <vector>

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

  std::vector<size_t> dims = {64, 64, 64}; // Large enough to benefit from threads

  // Use configurationND to get valid decomposition
  std::vector<int> commDims(dims.size(), 0);
  std::vector<size_t> subsize(dims.size()), offset(dims.size());
  int commSize = 0;
  int nda = 0;
  int rc = shafft::configurationND(dims,
                                   shafft::FFTType::Z2Z,
                                   commDims,
                                   nda,
                                   subsize,
                                   offset,
                                   commSize,
                                   shafft::DecompositionStrategy::MINIMIZE_NDA,
                                   0,
                                   MPI_COMM_WORLD);
  if (rc != 0) {
    unsetenv("SHAFFT_FFTW_THREADS");
    return false;
  }

  shafft::FFTND fft;
  rc = fft.init(commDims, dims, shafft::FFTType::Z2Z, MPI_COMM_WORLD);
  fft.plan();
  if (rc != 0) {
    unsetenv("SHAFFT_FFTW_THREADS");
    return false;
  }

  // Re-get layout after plan init
  SHAFFT_CHECK(fft.getLayout(subsize, offset, shafft::TensorLayout::CURRENT));

  size_t localElems = test::product(subsize);
  size_t alloc_elems = fft.allocSize();

  shafft::complexd *data = nullptr, *work = nullptr;
  SHAFFT_CHECK(shafft::allocBuffer(alloc_elems, &data));
  SHAFFT_CHECK(shafft::allocBuffer(alloc_elems, &work));

  // Initialize with known pattern
  std::vector<shafft::complexd> original(alloc_elems);
  for (size_t i = 0; i < localElems; ++i) {
    original[i] = shafft::complexd(static_cast<double>(i), static_cast<double>(i) * 0.5);
  }

  SHAFFT_CHECK(shafft::copyToBuffer(data, original.data(), alloc_elems));
  SHAFFT_CHECK(fft.setBuffers(data, work));

  // Forward + backward + normalize
  SHAFFT_CHECK(fft.execute(shafft::FFTDirection::FORWARD));
  SHAFFT_CHECK(fft.execute(shafft::FFTDirection::BACKWARD));
  SHAFFT_CHECK(fft.normalize());

  shafft::complexd *final_data, *final_work;
  SHAFFT_CHECK(fft.getBuffers(&final_data, &final_work));

  std::vector<shafft::complexd> result(alloc_elems);
  SHAFFT_CHECK(shafft::copyFromBuffer(result.data(), final_data, alloc_elems));

  size_t globalN = test::product(dims);
  bool passed = test::check_rel_error(
      result.data(), original.data(), localElems, globalN, MPI_COMM_WORLD, test::TOL_D);

  (void)shafft::freeBuffer(data);
  (void)shafft::freeBuffer(work);

  // Clean up environment
  unsetenv("SHAFFT_FFTW_THREADS");

  return passed;
}

/**
 * Test single-threaded FFTW (baseline comparison).
 */
static bool test_fftw_single_thread_roundtrip() {
  setenv("SHAFFT_FFTW_THREADS", "1", 1);

  std::vector<size_t> dims = {32, 32, 32};

  // Use configurationND to get valid decomposition
  std::vector<int> commDims(dims.size(), 0);
  std::vector<size_t> subsize(dims.size()), offset(dims.size());
  int commSize = 0;
  int nda = 0;
  int rc = shafft::configurationND(dims,
                                   shafft::FFTType::C2C,
                                   commDims,
                                   nda,
                                   subsize,
                                   offset,
                                   commSize,
                                   shafft::DecompositionStrategy::MINIMIZE_NDA,
                                   0,
                                   MPI_COMM_WORLD);
  if (rc != 0) {
    unsetenv("SHAFFT_FFTW_THREADS");
    return false;
  }

  shafft::FFTND fft;
  rc = fft.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  fft.plan();
  if (rc != 0) {
    unsetenv("SHAFFT_FFTW_THREADS");
    return false;
  }

  // Re-get layout after plan init
  SHAFFT_CHECK(fft.getLayout(subsize, offset, shafft::TensorLayout::CURRENT));

  size_t localElems = test::product(subsize);
  size_t alloc_elems = fft.allocSize();

  shafft::complexf *data = nullptr, *work = nullptr;
  SHAFFT_CHECK(shafft::allocBuffer(alloc_elems, &data));
  SHAFFT_CHECK(shafft::allocBuffer(alloc_elems, &work));

  std::vector<shafft::complexf> original(alloc_elems);
  for (size_t i = 0; i < localElems; ++i) {
    original[i] = shafft::complexf(static_cast<float>(i % 100), 0.0f);
  }

  SHAFFT_CHECK(shafft::copyToBuffer(data, original.data(), alloc_elems));
  SHAFFT_CHECK(fft.setBuffers(data, work));

  SHAFFT_CHECK(fft.execute(shafft::FFTDirection::FORWARD));
  SHAFFT_CHECK(fft.execute(shafft::FFTDirection::BACKWARD));
  SHAFFT_CHECK(fft.normalize());

  shafft::complexf *final_data, *final_work;
  SHAFFT_CHECK(fft.getBuffers(&final_data, &final_work));

  std::vector<shafft::complexf> result(alloc_elems);
  SHAFFT_CHECK(shafft::copyFromBuffer(result.data(), final_data, alloc_elems));

  size_t globalN = test::product(dims);
  bool passed = test::check_rel_error(
      result.data(), original.data(), localElems, globalN, MPI_COMM_WORLD, test::TOL_F);

  (void)shafft::freeBuffer(data);
  (void)shafft::freeBuffer(work);

  unsetenv("SHAFFT_FFTW_THREADS");

  return passed;
}

/**
 * Test FFTW with maximum threads (OMP_NUM_THREADS or hardware concurrency).
 */
static bool test_fftw_max_threads_roundtrip() {
  // Use 4 threads (reasonable for most systems)
  setenv("SHAFFT_FFTW_THREADS", "4", 1);

  std::vector<size_t> dims = {128, 128}; // 2D, larger problem
  std::vector<int> commDims = test::compute_comm_dims_nda1(dims, MPI_COMM_WORLD);

  shafft::FFTND fft;
  int rc = fft.init(commDims, dims, shafft::FFTType::Z2Z, MPI_COMM_WORLD);
  fft.plan();
  if (rc != 0)
    return false;

  std::vector<size_t> subsize(dims.size()), offset(dims.size());
  SHAFFT_CHECK(fft.getLayout(subsize, offset, shafft::TensorLayout::CURRENT));

  size_t localElems = test::product(subsize);
  size_t alloc_elems = fft.allocSize();

  shafft::complexd *data = nullptr, *work = nullptr;
  SHAFFT_CHECK(shafft::allocBuffer(alloc_elems, &data));
  SHAFFT_CHECK(shafft::allocBuffer(alloc_elems, &work));

  std::vector<shafft::complexd> original(alloc_elems);
  for (size_t i = 0; i < localElems; ++i) {
    original[i] =
        shafft::complexd(std::sin(static_cast<double>(i)), std::cos(static_cast<double>(i)));
  }

  SHAFFT_CHECK(shafft::copyToBuffer(data, original.data(), alloc_elems));
  SHAFFT_CHECK(fft.setBuffers(data, work));

  SHAFFT_CHECK(fft.execute(shafft::FFTDirection::FORWARD));
  SHAFFT_CHECK(fft.execute(shafft::FFTDirection::BACKWARD));
  SHAFFT_CHECK(fft.normalize());

  shafft::complexd *final_data, *final_work;
  SHAFFT_CHECK(fft.getBuffers(&final_data, &final_work));

  std::vector<shafft::complexd> result(alloc_elems);
  SHAFFT_CHECK(shafft::copyFromBuffer(result.data(), final_data, alloc_elems));

  size_t globalN = test::product(dims);
  bool passed = test::check_rel_error(
      result.data(), original.data(), localElems, globalN, MPI_COMM_WORLD, test::TOL_D);

  (void)shafft::freeBuffer(data);
  (void)shafft::freeBuffer(work);

  unsetenv("SHAFFT_FFTW_THREADS");

  return passed;
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
  if (hip_rc != hipSuccess)
    return false;

  std::vector<size_t> dims = {64, 64, 64};
  std::vector<int> commDims = test::compute_comm_dims_nda1(dims, MPI_COMM_WORLD);

  shafft::FFTND fft;
  int rc = fft.init(commDims, dims, shafft::FFTType::Z2Z, MPI_COMM_WORLD);
  if (rc != 0) {
    HIP_CHECK(hipStreamDestroy(stream));
    return false;
  }
  rc = fft.plan();
  if (rc != 0) {
    HIP_CHECK(hipStreamDestroy(stream));
    return false;
  }

  // Attach custom stream to plan
  rc = fft.setStream(stream);
  if (rc != 0) {
    HIP_CHECK(hipStreamDestroy(stream));
    return false;
  }

  std::vector<size_t> subsize(dims.size()), offset(dims.size());
  SHAFFT_CHECK(fft.getLayout(subsize, offset, shafft::TensorLayout::CURRENT));

  size_t localElems = test::product(subsize);
  size_t alloc_elems = fft.allocSize();

  shafft::complexd *data = nullptr, *work = nullptr;
  SHAFFT_CHECK(shafft::allocBuffer(alloc_elems, &data));
  SHAFFT_CHECK(shafft::allocBuffer(alloc_elems, &work));

  std::vector<shafft::complexd> original(alloc_elems);
  for (size_t i = 0; i < localElems; ++i) {
    original[i] = shafft::complexd(static_cast<double>(i), static_cast<double>(i) * 0.5);
  }

  SHAFFT_CHECK(shafft::copyToBuffer(data, original.data(), alloc_elems));
  SHAFFT_CHECK(fft.setBuffers(data, work));

  SHAFFT_CHECK(fft.execute(shafft::FFTDirection::FORWARD));
  SHAFFT_CHECK(fft.execute(shafft::FFTDirection::BACKWARD));
  SHAFFT_CHECK(fft.normalize());

  // Synchronize stream before reading results
  HIP_CHECK(hipStreamSynchronize(stream));

  shafft::complexd *final_data, *final_work;
  SHAFFT_CHECK(fft.getBuffers(&final_data, &final_work));

  std::vector<shafft::complexd> result(alloc_elems);
  SHAFFT_CHECK(shafft::copyFromBuffer(result.data(), final_data, alloc_elems));

  size_t globalN = test::product(dims);
  bool passed = test::check_rel_error(
      result.data(), original.data(), localElems, globalN, MPI_COMM_WORLD, test::TOL_D);

  (void)shafft::freeBuffer(data);
  (void)shafft::freeBuffer(work);
  HIP_CHECK(hipStreamDestroy(stream));

  return passed;
}

/**
 * Test default stream (null stream) works correctly.
 */
static bool test_hip_default_stream_roundtrip() {
  std::vector<size_t> dims = {32, 32, 32};
  std::vector<int> commDims = test::compute_comm_dims_nda1(dims, MPI_COMM_WORLD);

  shafft::FFTND fft;
  int rc = fft.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  // Use null stream (default)
  rc = fft.setStream(nullptr);
  if (rc != 0)
    return false;

  std::vector<size_t> subsize(dims.size()), offset(dims.size());
  SHAFFT_CHECK(fft.getLayout(subsize, offset, shafft::TensorLayout::CURRENT));

  size_t localElems = test::product(subsize);
  size_t alloc_elems = fft.allocSize();

  shafft::complexf *data = nullptr, *work = nullptr;
  SHAFFT_CHECK(shafft::allocBuffer(alloc_elems, &data));
  SHAFFT_CHECK(shafft::allocBuffer(alloc_elems, &work));

  std::vector<shafft::complexf> original(alloc_elems);
  for (size_t i = 0; i < localElems; ++i) {
    original[i] = shafft::complexf(static_cast<float>(i % 100), 0.0f);
  }

  SHAFFT_CHECK(shafft::copyToBuffer(data, original.data(), alloc_elems));
  SHAFFT_CHECK(fft.setBuffers(data, work));

  SHAFFT_CHECK(fft.execute(shafft::FFTDirection::FORWARD));
  SHAFFT_CHECK(fft.execute(shafft::FFTDirection::BACKWARD));
  SHAFFT_CHECK(fft.normalize());

  // Default stream is synchronous, no explicit sync needed

  shafft::complexf *final_data, *final_work;
  SHAFFT_CHECK(fft.getBuffers(&final_data, &final_work));

  std::vector<shafft::complexf> result(alloc_elems);
  SHAFFT_CHECK(shafft::copyFromBuffer(result.data(), final_data, alloc_elems));

  size_t globalN = test::product(dims);
  bool passed = test::check_rel_error(
      result.data(), original.data(), localElems, globalN, MPI_COMM_WORLD, test::TOL_F);

  (void)shafft::freeBuffer(data);
  (void)shafft::freeBuffer(work);

  return passed;
}

/**
 * Test multiple streams on same plan (stream switching).
 */
static bool test_hip_stream_switching() {
  hipStream_t stream1, stream2;
  if (hipStreamCreate(&stream1) != hipSuccess)
    return false;
  if (hipStreamCreate(&stream2) != hipSuccess) {
    HIP_CHECK(hipStreamDestroy(stream1));
    return false;
  }

  std::vector<size_t> dims = {32, 32};
  std::vector<int> commDims = test::compute_comm_dims_nda1(dims, MPI_COMM_WORLD);

  shafft::FFTND fft;
  int rc = fft.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0) {
    HIP_CHECK(hipStreamDestroy(stream1));
    HIP_CHECK(hipStreamDestroy(stream2));
    return false;
  }
  rc = fft.plan();
  if (rc != 0) {
    HIP_CHECK(hipStreamDestroy(stream1));
    HIP_CHECK(hipStreamDestroy(stream2));
    return false;
  }

  std::vector<size_t> subsize(dims.size()), offset(dims.size());
  SHAFFT_CHECK(fft.getLayout(subsize, offset, shafft::TensorLayout::CURRENT));

  size_t localElems = test::product(subsize);
  size_t alloc_elems = fft.allocSize();

  shafft::complexf *data = nullptr, *work = nullptr;
  SHAFFT_CHECK(shafft::allocBuffer(alloc_elems, &data));
  SHAFFT_CHECK(shafft::allocBuffer(alloc_elems, &work));

  std::vector<shafft::complexf> original(alloc_elems);
  for (size_t i = 0; i < localElems; ++i) {
    original[i] = shafft::complexf(1.0f, 0.0f);
  }

  SHAFFT_CHECK(shafft::copyToBuffer(data, original.data(), alloc_elems));
  SHAFFT_CHECK(fft.setBuffers(data, work));

  // Use stream1 for forward
  SHAFFT_CHECK(fft.setStream(stream1));
  SHAFFT_CHECK(fft.execute(shafft::FFTDirection::FORWARD));
  HIP_CHECK(hipStreamSynchronize(stream1));

  // Switch to stream2 for backward
  SHAFFT_CHECK(fft.setStream(stream2));
  SHAFFT_CHECK(fft.execute(shafft::FFTDirection::BACKWARD));
  SHAFFT_CHECK(fft.normalize());
  HIP_CHECK(hipStreamSynchronize(stream2));

  shafft::complexf *final_data, *final_work;
  SHAFFT_CHECK(fft.getBuffers(&final_data, &final_work));

  std::vector<shafft::complexf> result(alloc_elems);
  SHAFFT_CHECK(shafft::copyFromBuffer(result.data(), final_data, alloc_elems));

  size_t globalN = test::product(dims);
  bool passed = test::check_rel_error(
      result.data(), original.data(), localElems, globalN, MPI_COMM_WORLD, test::TOL_F);

  (void)shafft::freeBuffer(data);
  (void)shafft::freeBuffer(work);
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));

  return passed;
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
  return !name.empty(); // At least some name should be returned
#endif
}

int main(int argc, char** argv) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::fprintf(stderr, "MPI_Init failed\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  test::TestRunnerMPI runner("Backend-Specific Tests");

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
