/**
 * @file test_delta.cpp
 * @brief Test FFT of delta function produces expected constant output
 */
#include "test_utils.hpp"
#include <cmath>
#include <mpi.h>
#include <shafft/shafft.hpp>
#include <vector>

// Test that FFT of delta at origin gives constant value
static bool test_delta_3d() {
  std::vector<size_t> dims = {64, 64, 32};

  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  std::vector<int> commDims(dims.size(), 1);
  commDims[0] = worldSize;

  shafft::FFTND fft;
  int rc = fft.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  std::vector<size_t> subsize(3), offset(3);
  SHAFFT_CHECK(fft.getLayout(subsize, offset, shafft::TensorLayout::CURRENT));

  size_t alloc_elems = fft.allocSize();
  if (alloc_elems == 0)
    return false;

  shafft::complexf *data = nullptr, *work = nullptr;
  SHAFFT_CHECK(shafft::allocBuffer(alloc_elems, &data));
  SHAFFT_CHECK(shafft::allocBuffer(alloc_elems, &work));

  // Initialize: delta at origin
  std::vector<shafft::complexf> host(alloc_elems, {0.0f, 0.0f});

  // Check if origin is on this rank
  bool has_origin = (offset[0] == 0 && offset[1] == 0 && offset[2] == 0);
  if (has_origin) {
    host[0] = {1.0f, 0.0f};
  }

  SHAFFT_CHECK(shafft::copyToBuffer(data, host.data(), alloc_elems));
  SHAFFT_CHECK(fft.setBuffers(data, work));

  // Forward FFT
  rc = fft.execute(shafft::FFTDirection::FORWARD);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  // Normalize
  rc = fft.normalize();
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  // Get result
  shafft::complexf *result_data, *result_work;
  SHAFFT_CHECK(fft.getBuffers(&result_data, &result_work));

  std::vector<shafft::complexf> result(alloc_elems);
  SHAFFT_CHECK(shafft::copyFromBuffer(result.data(), result_data, alloc_elems));

  // Expected: constant value 1/sqrt(N) everywhere
  size_t N = test::product(dims);
  float expected_val = 1.0f / std::sqrt(static_cast<float>(N));
  shafft::complexf expected = {expected_val, 0.0f};

  // Get redistributed layout
  std::vector<size_t> trans_subsize(3), trans_offset(3);
  SHAFFT_CHECK(fft.getLayout(trans_subsize, trans_offset, shafft::TensorLayout::CURRENT));
  size_t trans_local = test::product(trans_subsize);

  bool passed = true;
  for (size_t i = 0; i < trans_local; ++i) {
    if (!test::approx_eq(result[i], expected, 1e-4f)) {
      passed = false;
      break;
    }
  }

  (void)shafft::freeBuffer(data);
  (void)shafft::freeBuffer(work);

  return passed;
}

// Test unnormalized FFT of delta gives all ones
static bool test_delta_unnormalized() {
  std::vector<size_t> dims = {32, 32};

  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  std::vector<int> commDims(dims.size(), 1);
  commDims[0] = worldSize;

  shafft::FFTND fft;
  int rc = fft.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  std::vector<size_t> subsize(2), offset(2);
  SHAFFT_CHECK(fft.getLayout(subsize, offset, shafft::TensorLayout::CURRENT));

  size_t alloc_elems = fft.allocSize();

  shafft::complexf *data = nullptr, *work = nullptr;
  SHAFFT_CHECK(shafft::allocBuffer(alloc_elems, &data));
  SHAFFT_CHECK(shafft::allocBuffer(alloc_elems, &work));

  // Delta at origin
  std::vector<shafft::complexf> host(alloc_elems, {0.0f, 0.0f});
  bool has_origin = (offset[0] == 0 && offset[1] == 0);
  if (has_origin)
    host[0] = {1.0f, 0.0f};

  SHAFFT_CHECK(shafft::copyToBuffer(data, host.data(), alloc_elems));
  SHAFFT_CHECK(fft.setBuffers(data, work));

  // Forward FFT without normalize
  rc = fft.execute(shafft::FFTDirection::FORWARD);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  shafft::complexf *result_data, *result_work;
  SHAFFT_CHECK(fft.getBuffers(&result_data, &result_work));

  std::vector<shafft::complexf> result(alloc_elems);
  SHAFFT_CHECK(shafft::copyFromBuffer(result.data(), result_data, alloc_elems));

  std::vector<size_t> trans_subsize(2), trans_offset(2);
  SHAFFT_CHECK(fft.getLayout(trans_subsize, trans_offset, shafft::TensorLayout::CURRENT));
  size_t trans_local = test::product(trans_subsize);

  // Expected: all ones (unnormalized)
  shafft::complexf expected = {1.0f, 0.0f};

  bool passed = true;
  for (size_t i = 0; i < trans_local; ++i) {
    if (!test::approx_eq(result[i], expected, 1e-5f)) {
      passed = false;
      break;
    }
  }

  (void)shafft::freeBuffer(data);
  (void)shafft::freeBuffer(work);

  return passed;
}

int main(int argc, char** argv) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::fprintf(stderr, "MPI_Init failed\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  test::TestRunnerMPI runner("Delta Function Tests");

  runner.run("3D delta -> constant (normalized)", test_delta_3d);
  runner.run("2D delta -> ones (unnormalized)", test_delta_unnormalized);

  int result = runner.finalize();
  MPI_Finalize();
  return result;
}
