/**
 * @file test_roundtrip.cpp
 * @brief Test that forward + backward + normalize recovers original data
 */
#include "test_utils.hpp"
#include <cmath>
#include <mpi.h>
#include <shafft/shafft.hpp>
#include <vector>

// Test round-trip for a given dimension configuration
static bool test_roundtrip_dims(const std::vector<size_t>& dims, int nda) {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  // Create commDims based on nda
  std::vector<int> commDims(dims.size(), 1);
  if (nda == 1) {
    commDims[0] = worldSize;
  } else if (nda == 2) {
    std::vector<int> mpi_dims(2, 0);
    MPI_Dims_create(worldSize, 2, mpi_dims.data());
    commDims[0] = mpi_dims[0];
    commDims[1] = mpi_dims[1];
  }

  shafft::FFTND fft;
  int rc = fft.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  // Get layout
  std::vector<size_t> subsize(dims.size()), offset(dims.size());
  SHAFFT_CHECK(fft.getLayout(subsize, offset, shafft::TensorLayout::CURRENT));

  size_t localElems = test::product(subsize);
  size_t alloc_elems = fft.allocSize();
  if (alloc_elems == 0)
    return false;

  // Allocate
  shafft::complexf *data = nullptr, *work = nullptr;
  SHAFFT_CHECK(shafft::allocBuffer(alloc_elems, &data));
  SHAFFT_CHECK(shafft::allocBuffer(alloc_elems, &work));

  // Initialize with pattern based on global position
  std::vector<shafft::complexf> original(alloc_elems, {0.0f, 0.0f});

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Simple pattern: delta at origin
  for (size_t i = 0; i < localElems; ++i) {
    // Check if this is the origin
    bool is_origin = true;
    size_t idx = i;
    for (int d = (int)dims.size() - 1; d >= 0; --d) {
      size_t local_coord = idx % subsize[d];
      size_t global_coord = offset[d] + local_coord;
      if (global_coord != 0)
        is_origin = false;
      idx /= subsize[d];
    }
    original[i] = is_origin ? shafft::complexf{1.0f, 0.0f} : shafft::complexf{0.0f, 0.0f};
  }

  SHAFFT_CHECK(shafft::copyToBuffer(data, original.data(), alloc_elems));
  SHAFFT_CHECK(fft.setBuffers(data, work));

  // Forward
  rc = fft.execute(shafft::FFTDirection::FORWARD);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  // Normalize after forward (symmetric scaling)
  rc = fft.normalize();
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  // Backward
  rc = fft.execute(shafft::FFTDirection::BACKWARD);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  // Normalize after backward
  rc = fft.normalize();
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  // Get final data
  shafft::complexf *final_data, *final_work;
  SHAFFT_CHECK(fft.getBuffers(&final_data, &final_work));

  // Verify layout is back to initial
  std::vector<size_t> final_subsize(dims.size()), final_offset(dims.size());
  SHAFFT_CHECK(fft.getLayout(final_subsize, final_offset, shafft::TensorLayout::CURRENT));

  for (size_t d = 0; d < dims.size(); ++d) {
    if (final_subsize[d] != subsize[d] || final_offset[d] != offset[d]) {
      (void)shafft::freeBuffer(data);
      (void)shafft::freeBuffer(work);
      return false;
    }
  }

  // Copy back and compare
  std::vector<shafft::complexf> result(alloc_elems);
  SHAFFT_CHECK(shafft::copyFromBuffer(result.data(), final_data, alloc_elems));

  bool passed = true;
  for (size_t i = 0; i < localElems; ++i) {
    if (!test::approx_eq(result[i], original[i], 1e-4f)) {
      passed = false;
      break;
    }
  }

  (void)shafft::freeBuffer(data);
  (void)shafft::freeBuffer(work);

  return passed;
}

// Test cases
static bool test_3d_nda1() {
  return test_roundtrip_dims({64, 64, 32}, 1);
}
static bool test_3d_nda2() {
  return test_roundtrip_dims({32, 32, 32}, 2);
}
static bool test_2d_nda1() {
  return test_roundtrip_dims({128, 64}, 1);
}
static bool test_4d_nda1() {
  return test_roundtrip_dims({16, 16, 16, 8}, 1);
}
static bool test_odd_dims() {
  return test_roundtrip_dims({63, 47, 31}, 1);
}

int main(int argc, char** argv) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::fprintf(stderr, "MPI_Init failed\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  test::TestRunnerMPI runner("Round-trip Tests");

  runner.run("3D NDA=1 [64x64x32]", test_3d_nda1);
  runner.run("3D NDA=2 [32x32x32]", test_3d_nda2);
  runner.run("2D NDA=1 [128x64]", test_2d_nda1);
  runner.run("4D NDA=1 [16x16x16x8]", test_4d_nda1);
  runner.run("Odd dimensions [63x47x31]", test_odd_dims);

  int result = runner.finalize();
  MPI_Finalize();
  return result;
}
