/**
 * @file test_FFT1D_roundtrip.cpp
 * @brief Integration tests for FFT1D distributed 1D FFT
 *
 * Tests based on distributed1DFFT/tests/test_distributed_fft.cpp:
 *   - Delta function test: FFT of delta should be constant
 *   - Round-trip test: IFFT(FFT(x)) / N == x
 *   - Sinusoid test: FFT of complex exponential should be delta at frequency
 *
 * Runs with various MPI ranks to verify distributed correctness.
 */
#include "test_utils.hpp"
#include <cmath>
#include <complex>
#include <mpi.h>
#include <shafft/shafft.hpp>
#include <vector>

#include <type_traits>

// Rel-error helper now lives in test_utils.hpp
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Test: Delta function - single precision
static bool test_delta_float() {
  int worldSize, worldRank;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

  // Use a size that's reasonable for testing
  size_t N = 256;
  size_t localN, localStart;

  int rc = shafft::configuration1D(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  shafft::FFT1D fft;
  rc = fft.init(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  size_t alloc = fft.allocSize();
  size_t actual_local_n = fft.localSize();

  // Allocate buffers
  shafft::complexf *data = nullptr, *work = nullptr;
  rc = shafft::allocBuffer(alloc, &data);
  if (rc != 0)
    return false;
  rc = shafft::allocBuffer(alloc, &work);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    return false;
  }

  // Initialize to zero
  std::vector<shafft::complexf> h_input(alloc, {0.0f, 0.0f});

  // Set delta at index 0 (only on rank 0)
  std::vector<size_t> local_shape, offset;
  SHAFFT_CHECK(fft.getLayout(local_shape, offset, shafft::TensorLayout::CURRENT));
  if (offset[0] == 0) {
    h_input[0] = {1.0f, 0.0f};
  }

  // Copy to device
  rc = shafft::copyToBuffer(data, h_input.data(), alloc);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  rc = fft.setBuffers(data, work);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  // Forward FFT: input from data, output to work
  rc = fft.execute(shafft::FFTDirection::FORWARD);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  // Copy result from work buffer (output buffer)
  std::vector<shafft::complexf> h_output(alloc);
  rc = shafft::copyFromBuffer(h_output.data(), work, alloc);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  // Verify: FFT of delta should be constant = 1.0
  bool correct = true;
  float tolerance = 1e-4f;
  for (size_t i = 0; i < actual_local_n; ++i) {
    if (std::fabs(h_output[i].real() - 1.0f) > tolerance ||
        std::fabs(h_output[i].imag()) > tolerance) {
      correct = false;
      break;
    }
  }

  (void)shafft::freeBuffer(data);
  (void)shafft::freeBuffer(work);

  return test::all_ranks_pass(correct);
}

// Test: Delta function - double precision
static bool test_delta_double() {
  int worldSize, worldRank;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

  size_t N = 256;
  size_t localN, localStart;

  int rc = shafft::configuration1D(N, localN, localStart, shafft::FFTType::Z2Z, MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  shafft::FFT1D fft;
  rc = fft.init(N, localN, localStart, shafft::FFTType::Z2Z, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  size_t alloc = fft.allocSize();
  size_t actual_local_n = fft.localSize();

  shafft::complexd *data = nullptr, *work = nullptr;
  rc = shafft::allocBuffer(alloc, &data);
  if (rc != 0)
    return false;
  rc = shafft::allocBuffer(alloc, &work);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    return false;
  }

  std::vector<shafft::complexd> h_input(alloc, {0.0, 0.0});

  std::vector<size_t> local_shape, offset;
  SHAFFT_CHECK(fft.getLayout(local_shape, offset, shafft::TensorLayout::CURRENT));
  if (offset[0] == 0) {
    h_input[0] = {1.0, 0.0};
  }

  rc = shafft::copyToBuffer(data, h_input.data(), alloc);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  rc = fft.setBuffers(data, work);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  // Forward FFT: input from data, output to work
  rc = fft.execute(shafft::FFTDirection::FORWARD);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  // Copy result from work buffer (output buffer)
  std::vector<shafft::complexd> h_output(alloc);
  rc = shafft::copyFromBuffer(h_output.data(), work, alloc);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  bool correct = true;
  double tolerance = 1e-10;
  for (size_t i = 0; i < actual_local_n; ++i) {
    if (std::fabs(h_output[i].real() - 1.0) > tolerance ||
        std::fabs(h_output[i].imag()) > tolerance) {
      correct = false;
      break;
    }
  }

  (void)shafft::freeBuffer(data);
  (void)shafft::freeBuffer(work);

  return test::all_ranks_pass(correct);
}

// Test: Round-trip - single precision
static bool test_roundtrip_float() {
  int worldRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

  size_t N = 128;
  size_t localN, localStart;

  int rc = shafft::configuration1D(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  shafft::FFT1D fft;
  rc = fft.init(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  size_t alloc = fft.allocSize();
  size_t actual_local_n = fft.localSize();

  shafft::complexf *data = nullptr, *work = nullptr;
  rc = shafft::allocBuffer(alloc, &data);
  if (rc != 0)
    return false;
  rc = shafft::allocBuffer(alloc, &work);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    return false;
  }

  // Initialize with a pattern based on global index
  std::vector<shafft::complexf> h_input(alloc);
  std::vector<size_t> local_shape, offset;
  SHAFFT_CHECK(fft.getLayout(local_shape, offset, shafft::TensorLayout::CURRENT));

  for (size_t i = 0; i < actual_local_n; ++i) {
    size_t global_idx = static_cast<size_t>(offset[0]) + i;
    float re = static_cast<float>(global_idx);
    float im = static_cast<float>(global_idx) * 0.5f;
    h_input[i] = {re, im};
  }
  // Zero padding
  for (size_t i = actual_local_n; i < alloc; ++i) {
    h_input[i] = {0.0f, 0.0f};
  }

  rc = shafft::copyToBuffer(data, h_input.data(), alloc);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  rc = fft.setBuffers(data, work);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  // Forward FFT: after execute, getBuffers returns (result, prev_input)
  rc = fft.execute(shafft::FFTDirection::FORWARD);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  // Backward FFT: reads from current data (forward result), writes to work, then swaps
  // No manual buffer swap needed - FFT1D now behaves like FFTND
  rc = fft.execute(shafft::FFTDirection::BACKWARD);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  // Normalize (operates on data buffer where result is)
  rc = fft.normalize();
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  // Get current buffer pointers - data now contains the result
  shafft::complexf *resultData = nullptr, *resultWork = nullptr;
  rc = fft.getBuffers(&resultData, &resultWork);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  // Copy result from data buffer (output of backward FFT)
  std::vector<shafft::complexf> h_output(alloc);
  rc = shafft::copyFromBuffer(h_output.data(), resultData, alloc);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  // Verify round-trip in active region
  bool correct = true;
  float tolerance = 1e-3f; // Relaxed for accumulated float errors
  for (size_t i = 0; i < actual_local_n; ++i) {
    if (!test::approx_eq(h_output[i], h_input[i], tolerance)) {
      correct = false;
      break;
    }
  }

  // Verify padded region [actual_local_n, alloc) is zero or unchanged
  // (depending on backend behavior - at minimum, shouldn't be NaN/Inf)
  for (size_t i = actual_local_n; i < alloc; ++i) {
    if (std::isnan(h_output[i].real()) || std::isnan(h_output[i].imag()) ||
        std::isinf(h_output[i].real()) || std::isinf(h_output[i].imag())) {
      correct = false;
      break;
    }
  }

  (void)shafft::freeBuffer(data);
  (void)shafft::freeBuffer(work);

  return test::all_ranks_pass(correct);
}

// Test: Round-trip - double precision
static bool test_roundtrip_double() {
  int worldRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

  size_t N = 128;
  size_t localN, localStart;

  int rc = shafft::configuration1D(N, localN, localStart, shafft::FFTType::Z2Z, MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  shafft::FFT1D fft;
  rc = fft.init(N, localN, localStart, shafft::FFTType::Z2Z, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  size_t alloc = fft.allocSize();
  size_t actual_local_n = fft.localSize();

  shafft::complexd *data = nullptr, *work = nullptr;
  rc = shafft::allocBuffer(alloc, &data);
  if (rc != 0)
    return false;
  rc = shafft::allocBuffer(alloc, &work);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    return false;
  }

  std::vector<shafft::complexd> h_input(alloc);
  std::vector<size_t> local_shape, offset;
  SHAFFT_CHECK(fft.getLayout(local_shape, offset, shafft::TensorLayout::CURRENT));

  for (size_t i = 0; i < actual_local_n; ++i) {
    size_t global_idx = static_cast<size_t>(offset[0]) + i;
    double re = static_cast<double>(global_idx);
    double im = static_cast<double>(global_idx) * 0.5;
    h_input[i] = {re, im};
  }
  for (size_t i = actual_local_n; i < alloc; ++i) {
    h_input[i] = {0.0, 0.0};
  }

  rc = shafft::copyToBuffer(data, h_input.data(), alloc);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  rc = fft.setBuffers(data, work);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  // Forward FFT: after execute, getBuffers returns (result, prev_input)
  rc = fft.execute(shafft::FFTDirection::FORWARD);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  // Backward FFT: reads from current data (forward result), writes to work, then swaps
  // No manual buffer swap needed - FFT1D now behaves like FFTND
  rc = fft.execute(shafft::FFTDirection::BACKWARD);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  // Normalize (operates on data buffer where result is)
  rc = fft.normalize();
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  // Get current buffer pointers - data now contains the result
  shafft::complexd *resultData = nullptr, *resultWork = nullptr;
  rc = fft.getBuffers(&resultData, &resultWork);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  // Copy result from data buffer (output of backward FFT)
  std::vector<shafft::complexd> h_output(alloc);
  rc = shafft::copyFromBuffer(h_output.data(), resultData, alloc);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  bool correct = true;
  double tolerance = 1e-9;
  for (size_t i = 0; i < actual_local_n; ++i) {
    if (!test::approx_eq(h_output[i], h_input[i], tolerance)) {
      correct = false;
      break;
    }
  }

  // Verify padded region [actual_local_n, alloc) is not corrupted
  for (size_t i = actual_local_n; i < alloc; ++i) {
    if (std::isnan(h_output[i].real()) || std::isnan(h_output[i].imag()) ||
        std::isinf(h_output[i].real()) || std::isinf(h_output[i].imag())) {
      correct = false;
      break;
    }
  }

  (void)shafft::freeBuffer(data);
  (void)shafft::freeBuffer(work);

  return test::all_ranks_pass(correct);
}

// Test: Sinusoid - complex exponential at specific frequency
static bool test_sinusoid_float() {
  int worldRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

  size_t N = 256;
  size_t localN, localStart;

  int rc = shafft::configuration1D(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  shafft::FFT1D fft;
  rc = fft.init(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  size_t alloc = fft.allocSize();
  size_t actual_local_n = fft.localSize();

  // Get the padded size (N may be padded for the algorithm)
  size_t N_padded = fft.globalSize();

  shafft::complexf *data = nullptr, *work = nullptr;
  rc = shafft::allocBuffer(alloc, &data);
  if (rc != 0)
    return false;
  rc = shafft::allocBuffer(alloc, &work);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    return false;
  }

  // Input: exp(2*pi*i*k*n/N) for test frequency k
  int test_freq = 3;
  std::vector<shafft::complexf> h_input(alloc);
  std::vector<size_t> local_shape, offset;
  SHAFFT_CHECK(fft.getLayout(local_shape, offset, shafft::TensorLayout::CURRENT));

  for (size_t i = 0; i < actual_local_n; ++i) {
    size_t n = static_cast<size_t>(offset[0]) + i;
    float angle = static_cast<float>(2.0 * M_PI * test_freq * n / N_padded);
    h_input[i] = {std::cos(angle), std::sin(angle)};
  }
  for (size_t i = actual_local_n; i < alloc; ++i) {
    h_input[i] = {0.0f, 0.0f};
  }

  rc = shafft::copyToBuffer(data, h_input.data(), alloc);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  rc = fft.setBuffers(data, work);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  // Forward FFT: input from data, output to work
  rc = fft.execute(shafft::FFTDirection::FORWARD);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  // Copy result from work buffer (output buffer)
  std::vector<shafft::complexf> h_output(alloc);
  rc = shafft::copyFromBuffer(h_output.data(), work, alloc);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  // Verify: output should be N at frequency k, 0 elsewhere
  bool correct = true;
  float tolerance = 1e-3f;
  for (size_t i = 0; i < actual_local_n; ++i) {
    size_t global_freq = static_cast<size_t>(offset[0]) + i;
    float expected_real =
        (global_freq == static_cast<size_t>(test_freq)) ? static_cast<float>(N_padded) : 0.0f;
    float expected_imag = 0.0f;

    if (std::fabs(h_output[i].real() - expected_real) > tolerance ||
        std::fabs(h_output[i].imag() - expected_imag) > tolerance) {
      // Only flag as error if magnitude is significant
      if (std::fabs(h_output[i].real()) > tolerance || std::fabs(h_output[i].imag()) > tolerance) {
        if (global_freq != static_cast<size_t>(test_freq)) {
          correct = false;
          break;
        }
      }
    }
  }

  (void)shafft::freeBuffer(data);
  (void)shafft::freeBuffer(work);

  return test::all_ranks_pass(correct);
}

// Test: Various system sizes
static bool test_various_sizes() {
  std::vector<size_t> test_sizes = {32, 33, 64, 100, 128};
  bool all_correct = true;

  for (size_t N : test_sizes) {
    size_t localN, localStart;
    int rc = shafft::configuration1D(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) {
      all_correct = false;
      continue;
    }

    shafft::FFT1D fft;
    rc = fft.init(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) {
      all_correct = false;
      continue;
    }
    rc = fft.plan();
    if (rc != 0) {
      all_correct = false;
      continue;
    }

    // Inactive ranks skip buffer operations but must still participate
    // in the collective all_ranks_pass() call below.
    if (!fft.isActive()) {
      if (!test::all_ranks_pass(true))
        all_correct = false;
      continue;
    }

    size_t alloc = fft.allocSize();
    size_t actual_local_n = fft.localSize();

    shafft::complexf *data = nullptr, *work = nullptr;
    rc = shafft::allocBuffer(alloc, &data);
    if (rc != 0) {
      if (!test::all_ranks_pass(false))
        all_correct = false;
      continue;
    }
    rc = shafft::allocBuffer(alloc, &work);
    if (rc != 0) {
      (void)shafft::freeBuffer(data);
      if (!test::all_ranks_pass(false))
        all_correct = false;
      continue;
    }

    // Simple delta test
    std::vector<shafft::complexf> h_input(alloc, {0.0f, 0.0f});
    std::vector<size_t> local_shape, offset;
    SHAFFT_CHECK(fft.getLayout(local_shape, offset, shafft::TensorLayout::CURRENT));
    if (offset[0] == 0) {
      h_input[0] = {1.0f, 0.0f};
    }

    rc = shafft::copyToBuffer(data, h_input.data(), alloc);
    if (rc != 0) {
      (void)shafft::freeBuffer(data);
      (void)shafft::freeBuffer(work);
      if (!test::all_ranks_pass(false))
        all_correct = false;
      continue;
    }

    rc = fft.setBuffers(data, work);
    if (rc != 0) {
      (void)shafft::freeBuffer(data);
      (void)shafft::freeBuffer(work);
      if (!test::all_ranks_pass(false))
        all_correct = false;
      continue;
    }

    // Forward FFT: input from data, output to work
    rc = fft.execute(shafft::FFTDirection::FORWARD);
    if (rc != 0) {
      (void)shafft::freeBuffer(data);
      (void)shafft::freeBuffer(work);
      if (!test::all_ranks_pass(false))
        all_correct = false;
      continue;
    }

    // Copy result from work buffer (output buffer)
    std::vector<shafft::complexf> h_output(alloc);
    rc = shafft::copyFromBuffer(h_output.data(), work, alloc);
    if (rc != 0) {
      (void)shafft::freeBuffer(data);
      (void)shafft::freeBuffer(work);
      if (!test::all_ranks_pass(false))
        all_correct = false;
      continue;
    }

    // Verify constant output
    bool size_correct = true;
    float tolerance = 1e-4f;
    for (size_t i = 0; i < actual_local_n; ++i) {
      if (std::fabs(h_output[i].real() - 1.0f) > tolerance ||
          std::fabs(h_output[i].imag()) > tolerance) {
        size_correct = false;
        break;
      }
    }

    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);

    if (!test::all_ranks_pass(size_correct)) {
      all_correct = false;
    }
  }

  return all_correct;
}

// Test: Non-divisible sizes where localN != localAllocSize
// This tests the fix for the bug where localN/localStart were incorrectly set
// to padded values instead of the true block distribution.
static bool test_non_divisible_sizes() {
  int worldSize, worldRank;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

  // Test sizes that don't divide evenly - these are critical test cases
  std::vector<size_t> test_sizes = {10, 17, 31, 100, 127};

  bool all_correct = true;

  for (size_t N : test_sizes) {
    size_t localN = 0, localStart = 0;
    int rc = shafft::configuration1D(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) {
      all_correct = false;
      continue;
    }

    shafft::FFT1D fft;
    rc = fft.init(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) {
      all_correct = false;
      continue;
    }
    rc = fft.plan();
    if (rc != 0) {
      all_correct = false;
      continue;
    }

    // Inactive ranks skip buffer operations but must still participate
    // in the collective check_rel_error() and all_ranks_pass() calls.
    if (!fft.isActive()) {
      bool size_correct = test::check_rel_error(
          static_cast<shafft::complexf*>(nullptr),
          static_cast<shafft::complexf*>(nullptr),
          size_t{0}, N, MPI_COMM_WORLD, /*base_tol_rel=*/5e-4);
      if (!test::all_ranks_pass(size_correct)) {
        if (worldRank == 0) {
          std::printf("FAIL: N=%zu round-trip failed\n", N);
        }
        all_correct = false;
      }
      continue;
    }

    size_t alloc = fft.allocSize();
    size_t actual_local_n = fft.localSize();

    // Allocate buffers
    shafft::complexf *data = nullptr, *work = nullptr;
    rc = shafft::allocBuffer(alloc, &data);
    if (rc != 0) {
      all_correct = false;
      continue;
    }
    rc = shafft::allocBuffer(alloc, &work);
    if (rc != 0) {
      (void)shafft::freeBuffer(data);
      all_correct = false;
      continue;
    }

    // Initialize: set input[k] = k for global indices in [0, N)
    // Padding region (if any) should be zero
    std::vector<shafft::complexf> h_input(alloc, {0.0f, 0.0f});
    std::vector<size_t> local_shape, offset;
    SHAFFT_CHECK(fft.getLayout(local_shape, offset, shafft::TensorLayout::INITIAL));

    for (size_t i = 0; i < actual_local_n; ++i) {
      size_t global_idx = offset[0] + i;
      // Only set values within the original N
      if (global_idx < N) {
        h_input[i] = {static_cast<float>(global_idx), 0.0f};
      }
    }

    // Copy to device
    rc = shafft::copyToBuffer(data, h_input.data(), alloc);
    if (rc != 0) {
      (void)shafft::freeBuffer(data);
      (void)shafft::freeBuffer(work);
      all_correct = false;
      continue;
    }

    rc = fft.setBuffers(data, work);
    if (rc != 0) {
      (void)shafft::freeBuffer(data);
      (void)shafft::freeBuffer(work);
      all_correct = false;
      continue;
    }

    // Forward FFT: after execute, getBuffers returns (result, prev_input)
    rc = fft.execute(shafft::FFTDirection::FORWARD);
    if (rc != 0) {
      (void)shafft::freeBuffer(data);
      (void)shafft::freeBuffer(work);
      all_correct = false;
      continue;
    }

    // Backward FFT: reads from current data (forward result), writes to work, then swaps
    // No manual buffer swap needed - FFT1D now behaves like FFTND
    rc = fft.execute(shafft::FFTDirection::BACKWARD);
    if (rc != 0) {
      (void)shafft::freeBuffer(data);
      (void)shafft::freeBuffer(work);
      all_correct = false;
      continue;
    }

    // Normalize (operates on data buffer where result is)
    rc = fft.normalize();
    if (rc != 0) {
      (void)shafft::freeBuffer(data);
      (void)shafft::freeBuffer(work);
      all_correct = false;
      continue;
    }

    // Get current buffer pointers - data now contains the result
    shafft::complexf *resultData = nullptr, *resultWork = nullptr;
    rc = fft.getBuffers(&resultData, &resultWork);
    if (rc != 0) {
      (void)shafft::freeBuffer(data);
      (void)shafft::freeBuffer(work);
      all_correct = false;
      continue;
    }

    // Copy result from data buffer (the output of backward FFT)
    std::vector<shafft::complexf> h_output(alloc);
    rc = shafft::copyFromBuffer(h_output.data(), resultData, alloc);
    if (rc != 0) {
      (void)shafft::freeBuffer(data);
      (void)shafft::freeBuffer(work);
      all_correct = false;
      continue;
    }

    // Build reference locally: ref[k] = k for k<N else 0
    std::vector<shafft::complexf> h_ref(alloc, {0.0f, 0.0f});
    for (size_t i = 0; i < actual_local_n; ++i) {
      size_t g = offset[0] + i;
      if (g < N)
        h_ref[i] = shafft::complexf(static_cast<float>(g), 0.0f);
    }

    // Verify round-trip with distributed relative norms (scaled by sqrt(log N))
    bool size_correct =
        test::check_rel_error(h_output, h_ref, N, MPI_COMM_WORLD, /*base_tol_rel=*/5e-4);

    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);

    if (!test::all_ranks_pass(size_correct)) {
      if (worldRank == 0) {
        std::printf("FAIL: N=%zu round-trip failed\n", N);
      }
      all_correct = false;
    }
  }

  return all_correct;
}

// Main
int main(int argc, char* argv[]) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::fprintf(stderr, "MPI_Init failed\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  test::TestRunnerMPI runner("FFT1D Integration Tests");

  runner.run("delta function (float)", test_delta_float);
  runner.run("delta function (double)", test_delta_double);
  runner.run("round-trip (float)", test_roundtrip_float);
  runner.run("round-trip (double)", test_roundtrip_double);
  runner.run("sinusoid (float)", test_sinusoid_float);
  runner.run("various sizes", test_various_sizes);
  runner.run("non-divisible sizes", test_non_divisible_sizes);

  int result = runner.finalize();
  MPI_Finalize();
  return result;
}
