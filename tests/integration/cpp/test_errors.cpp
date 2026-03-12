/**
 * @file test_errors.cpp
 * @brief Test that invalid inputs return correct error codes
 */
#include "test_utils.hpp"
#include <mpi.h>
#include <shafft/shafft.hpp>
#include <vector>

// Helper to convert Status enum to int for comparison
constexpr int S(shafft::Status s) {
  return static_cast<int>(s);
}

// Test that 1D tensor with nda > 0 returns error (only with multi-rank)
// Note: With single rank, nda is forced to 0, so this test only makes sense with 2+ ranks
static bool test_1d_invalid_nda() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  // With single rank, the library allows any nda (forces to 0)
  // This test is only meaningful with multiple ranks
  if (worldSize == 1)
    return true; // Skip for single rank

  shafft::FFTND fft;
  std::vector<size_t> dims = {64}; // 1D

  // commDims = {worldSize} means nda=1, which should fail for 1D
  std::vector<int> commDims = {worldSize};
  int rc = fft.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0) {
    // init itself may reject invalid decomposition
    return rc == S(shafft::Status::ERR_INVALID_DECOMP);
  }
  rc = fft.plan();

  // Should return SHAFFT_ERR_INVALID_DECOMP
  return rc == S(shafft::Status::ERR_INVALID_DECOMP);
}

// Test that nda >= ndim returns error (need at least 1 contiguous axis)
static bool test_nda_too_large() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  // With single rank, nda is forced to 0
  if (worldSize == 1)
    return true;

  shafft::FFTND fft;
  std::vector<size_t> dims = {32, 32}; // 2D

  std::vector<int> commDims = {1, worldSize};
  int rc = fft.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0) {
    // init itself may reject invalid decomposition
    return rc == S(shafft::Status::ERR_INVALID_DECOMP);
  }
  rc = fft.plan();

  return rc == S(shafft::Status::ERR_INVALID_DECOMP);
}

// Test that zero dimension returns error
static bool test_zero_dim() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  shafft::FFTND fft;
  std::vector<size_t> dims = {64, 0, 32};
  std::vector<int> commDims = {worldSize, 1, 1};

  int rc = fft.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != S(shafft::Status::SUCCESS)) {
    // init rejected zero dim - that's acceptable
    return true;
  }
  rc = fft.plan();

  return rc != S(shafft::Status::SUCCESS);
}

// Test that invalid (overflow) dimension returns error
// Note: size_t can't be negative, but overflow creates huge values
static bool test_negative_dim() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  shafft::FFTND fft;
  // Using overflow to simulate "negative" - this creates a huge value
  std::vector<size_t> dims = {64, static_cast<size_t>(-1), 32};
  std::vector<int> commDims = {worldSize, 1, 1};

  int rc = fft.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != S(shafft::Status::SUCCESS)) {
    // init rejected overflow dim - that's acceptable
    return true;
  }
  rc = fft.plan();

  return rc != S(shafft::Status::SUCCESS);
}

// Test that empty dims returns error
static bool test_empty_dims() {
  shafft::FFTND fft;
  std::vector<size_t> dims = {};
  std::vector<int> commDims = {};

  int rc = fft.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != S(shafft::Status::SUCCESS)) {
    // init rejected empty dims - that's acceptable
    return true;
  }
  rc = fft.plan();

  return rc != S(shafft::Status::SUCCESS);
}

// Test execute without setBuffers returns error
static bool test_execute_no_buffers() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  shafft::FFTND fft;
  std::vector<size_t> dims = {32, 32};
  std::vector<int> commDims = {worldSize, 1};

  int rc = fft.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != S(shafft::Status::SUCCESS))
    return false;
  rc = fft.plan();
  if (rc != S(shafft::Status::SUCCESS))
    return false;

  // Don't set buffers, try to execute
  rc = fft.execute(shafft::FFTDirection::FORWARD);

  return rc == S(shafft::Status::ERR_NO_BUFFER);
}

// Test setBuffers with null pointers
static bool test_null_buffers() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  shafft::FFTND fft;
  std::vector<size_t> dims = {32, 32};
  std::vector<int> commDims = {worldSize, 1};

  int rc = fft.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != S(shafft::Status::SUCCESS))
    return false;
  rc = fft.plan();
  if (rc != S(shafft::Status::SUCCESS))
    return false;

  shafft::complexf* null_ptr = nullptr;
  rc = fft.setBuffers(null_ptr, null_ptr);

  return rc == S(shafft::Status::ERR_NULLPTR);
}

// Test double release is safe
static bool test_double_release() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  shafft::FFTND fft;
  std::vector<size_t> dims = {32, 32};
  std::vector<int> commDims = {worldSize, 1};

  int rc = fft.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != S(shafft::Status::SUCCESS))
    return false;
  rc = fft.plan();
  if (rc != S(shafft::Status::SUCCESS))
    return false;

  fft.release();
  fft.release(); // Should not crash

  return true;
}

int main(int argc, char** argv) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::fprintf(stderr, "MPI_Init failed\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  test::TestRunnerMPI runner("Error Handling Tests");

  runner.run("1D with nda=1 -> INVALID_DECOMP", test_1d_invalid_nda);
  runner.run("nda > ndim-1 -> INVALID_DECOMP", test_nda_too_large);
  runner.run("Zero dimension -> error", test_zero_dim);
  runner.run("Negative dimension -> error", test_negative_dim);
  runner.run("Empty dims -> error", test_empty_dims);
  runner.run("Execute without buffers -> NULLPTR", test_execute_no_buffers);
  runner.run("Null buffer pointers -> NULLPTR", test_null_buffers);
  runner.run("Double release is safe", test_double_release);

  int result = runner.finalize();
  MPI_Finalize();
  return result;
}
