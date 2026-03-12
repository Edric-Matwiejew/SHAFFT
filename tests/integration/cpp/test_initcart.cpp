/**
 * @file test_initcart.cpp
 * @brief Test init with explicit Cartesian process grid
 */
#include "test_utils.hpp"
#include <mpi.h>
#include <numeric>
#include <shafft/shafft.hpp>
#include <vector>

// Test: Basic init with matching grid
static bool test_basic_init() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  std::vector<size_t> dims = {64, 64, 32};
  // Cartesian grid: all ranks on first axis, 1 on others
  // Last axis must be 1 (not distributed)
  std::vector<int> commDims = {worldSize, 1, 1};

  shafft::FFTND fft;
  int rc = fft.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  rc = fft.plan();

  if (rc != 0)
    return false;
  if (!fft.isConfigured())
    return false;

  // Verify we can query the plan
  size_t alloc = fft.allocSize();
  return alloc > 0;
}

// Test: init with 2D grid (4 ranks = 2x2x1)
static bool test_2d_grid() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  if (worldSize != 4)
    return true; // Skip unless exactly 4 ranks

  std::vector<size_t> dims = {64, 64, 32};
  std::vector<int> commDims = {2, 2, 1}; // 2x2 grid on first two axes

  shafft::FFTND fft;
  int rc = fft.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();

  if (rc != 0)
    return false;

  // Verify layout - first two axes should be distributed
  std::vector<size_t> subsize(3), offset(3);
  SHAFFT_CHECK(fft.getLayout(subsize, offset, shafft::TensorLayout::INITIAL));

  // With 2x2 grid: each rank gets half of dim 0 and half of dim 1
  // But dim 2 should be full
  bool dim2_full = (subsize[2] == static_cast<size_t>(dims[2]));

  return dim2_full;
}

// Test: init produces consistent results
static bool test_init_consistent() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  std::vector<size_t> dims = {64, 64, 32};

  // Create two plans with same commDims
  std::vector<int> commDims = {worldSize, 1, 1};

  shafft::FFTND fft1;
  int rc1 = fft1.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc1 != 0)
    return false;

  shafft::FFTND fft2;
  int rc2 = fft2.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc2 != 0)
    return false;

  // Both should produce same allocSize
  size_t alloc1 = fft1.allocSize();
  size_t alloc2 = fft2.allocSize();

  if (alloc1 != alloc2)
    return false;

  // Both should produce same initial layout
  std::vector<size_t> sub1(3), off1(3);
  std::vector<size_t> sub2(3), off2(3);

  SHAFFT_CHECK(fft1.getLayout(sub1, off1, shafft::TensorLayout::INITIAL));
  SHAFFT_CHECK(fft2.getLayout(sub2, off2, shafft::TensorLayout::INITIAL));

  return (sub1 == sub2 && off1 == off2);
}

// Test: 2D tensor with init
static bool test_2d_init() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  std::vector<size_t> dims = {128, 64};
  std::vector<int> commDims = {worldSize, 1};

  shafft::FFTND fft;
  int rc = fft.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  rc = fft.plan();

  if (rc != 0)
    return false;

  std::vector<size_t> subsize(2), offset(2);
  SHAFFT_CHECK(fft.getLayout(subsize, offset, shafft::TensorLayout::INITIAL));

  // Second axis should be full
  return subsize[1] == static_cast<size_t>(dims[1]);
}

// Test: init with double precision
static bool test_init_double() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  std::vector<size_t> dims = {32, 32, 32};
  std::vector<int> commDims = {worldSize, 1, 1};

  shafft::FFTND fft;
  int rc = fft.init(commDims, dims, shafft::FFTType::Z2Z, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();

  if (rc != 0)
    return false;

  // Double precision should use twice the memory per element
  // but allocSize is in elements, so it should be same
  size_t alloc = fft.allocSize();

  return alloc > 0;
}

// Test: Execute works after init
static bool test_init_execute() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  std::vector<size_t> dims = {32, 32, 32};
  std::vector<int> commDims = {worldSize, 1, 1};

  shafft::FFTND fft;
  int rc = fft.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  size_t n = fft.allocSize();
  shafft::complexf *data, *work;
  SHAFFT_CHECK(shafft::allocBuffer(n, &data));
  SHAFFT_CHECK(shafft::allocBuffer(n, &work));

  std::vector<shafft::complexf> original(n);
  for (size_t i = 0; i < n; ++i) {
    original[i] = {static_cast<float>(i % 7 + 1), 0.0f};
  }
  SHAFFT_CHECK(shafft::copyToBuffer(data, original.data(), n));
  SHAFFT_CHECK(fft.setBuffers(data, work));

  // Forward + backward + normalize
  SHAFFT_CHECK(fft.execute(shafft::FFTDirection::FORWARD));
  SHAFFT_CHECK(fft.execute(shafft::FFTDirection::BACKWARD));
  SHAFFT_CHECK(fft.normalize());

  // Get result
  shafft::complexf *result_data, *result_work;
  SHAFFT_CHECK(fft.getBuffers(&result_data, &result_work));

  std::vector<shafft::complexf> result(n);
  SHAFFT_CHECK(shafft::copyFromBuffer(result.data(), result_data, n));

  // Verify recovery
  bool match = true;
  for (size_t i = 0; i < n; ++i) {
    if (!test::approx_eq(result[i], original[i], 1e-4f)) {
      match = false;
      break;
    }
  }

  (void)shafft::freeBuffer(data);
  (void)shafft::freeBuffer(work);

  return match;
}

// Main
int main(int argc, char** argv) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::fprintf(stderr, "MPI_Init failed\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  test::TestRunnerMPI runner("FFTND init Tests");

  runner.run("basic_init", test_basic_init);
  runner.run("2d_grid", test_2d_grid);
  runner.run("init_consistent", test_init_consistent);
  runner.run("2d_init", test_2d_init);
  runner.run("init_double", test_init_double);
  runner.run("init_execute", test_init_execute);

  int result = runner.finalize();
  MPI_Finalize();
  return result;
}
