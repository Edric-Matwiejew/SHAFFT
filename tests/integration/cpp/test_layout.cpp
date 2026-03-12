/**
 * @file test_layout.cpp
 * @brief Test that getLayout returns correct subsize/offset for various configurations
 *
 * Note: Basic subsize sum and offset contiguity tests are in test_configuration_modes.cpp.
 * This file focuses on Plan::getLayout() behavior with INITIAL/REDISTRIBUTED/CURRENT layouts.
 */
#include "test_utils.hpp"
#include <mpi.h>
#include <numeric>
#include <shafft/shafft.hpp>
#include <vector>

// Test REDISTRIBUTED layout is different from INITIAL for NDA >= 1
static bool test_redistributed_layout() {
  std::vector<size_t> dims = {64, 64, 32};

  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  if (worldSize < 2)
    return true; // Skip with single rank

  // Create commDims for nda=1
  std::vector<int> commDims(dims.size(), 1);
  commDims[0] = worldSize;

  shafft::FFTND fft;
  int rc = fft.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  std::vector<size_t> init_subsize(3), init_offset(3);
  std::vector<size_t> trans_subsize(3), trans_offset(3);

  SHAFFT_CHECK(fft.getLayout(init_subsize, init_offset, shafft::TensorLayout::INITIAL));
  SHAFFT_CHECK(fft.getLayout(trans_subsize, trans_offset, shafft::TensorLayout::REDISTRIBUTED));

  // For nda=1, distribution should shift from axis 0 to last axis
  // Initial: axis 0 is distributed (subsize[0] < dims[0])
  // Redistributed: last axis is distributed (subsize[2] < dims[2])

  bool init_dist_0 = (init_subsize[0] < dims[0]);
  bool trans_dist_2 = (trans_subsize[2] < dims[2]);

  return init_dist_0 && trans_dist_2;
}

// Test CURRENT tracks execute() calls
static bool test_current_tracks_state() {
  std::vector<size_t> dims = {32, 32, 32};

  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  if (worldSize < 2)
    return true;

  // Create commDims for nda=1
  std::vector<int> commDims(dims.size(), 1);
  commDims[0] = worldSize;

  shafft::FFTND fft;
  int rc = fft.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  // Before execute: CURRENT == INITIAL
  std::vector<size_t> init_sub(3), init_off(3);
  std::vector<size_t> curr_sub(3), curr_off(3);

  SHAFFT_CHECK(fft.getLayout(init_sub, init_off, shafft::TensorLayout::INITIAL));
  SHAFFT_CHECK(fft.getLayout(curr_sub, curr_off, shafft::TensorLayout::CURRENT));

  for (int i = 0; i < 3; ++i) {
    if (init_sub[i] != curr_sub[i] || init_off[i] != curr_off[i]) {
      return false;
    }
  }

  // Allocate and execute forward
  size_t n = fft.allocSize();
  shafft::complexf *data, *work;
  SHAFFT_CHECK(shafft::allocBuffer(n, &data));
  SHAFFT_CHECK(shafft::allocBuffer(n, &work));

  std::vector<shafft::complexf> host(n, {0.0f, 0.0f});
  SHAFFT_CHECK(shafft::copyToBuffer(data, host.data(), n));
  SHAFFT_CHECK(fft.setBuffers(data, work));

  SHAFFT_CHECK(fft.execute(shafft::FFTDirection::FORWARD));

  // After forward: CURRENT == REDISTRIBUTED
  std::vector<size_t> trans_sub(3), trans_off(3);
  SHAFFT_CHECK(fft.getLayout(trans_sub, trans_off, shafft::TensorLayout::REDISTRIBUTED));
  SHAFFT_CHECK(fft.getLayout(curr_sub, curr_off, shafft::TensorLayout::CURRENT));

  bool match_trans = true;
  for (int i = 0; i < 3; ++i) {
    if (trans_sub[i] != curr_sub[i] || trans_off[i] != curr_off[i]) {
      match_trans = false;
    }
  }

  (void)shafft::freeBuffer(data);
  (void)shafft::freeBuffer(work);

  return match_trans;
}

// Test policy INITIAL restores CURRENT layout to INITIAL after forward execute
static bool test_policy_initial_restores_layout() {
  std::vector<size_t> dims = {32, 32, 32};
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  if (worldSize < 2)
    return true;

  std::vector<int> commDims(dims.size(), 1);
  commDims[0] = worldSize;

  shafft::FFTND fft;
  int rc = fft.init(
      commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD, shafft::TransformLayout::INITIAL);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  size_t n = fft.allocSize();
  shafft::complexf *data, *work;
  SHAFFT_CHECK(shafft::allocBuffer(n, &data));
  SHAFFT_CHECK(shafft::allocBuffer(n, &work));
  std::vector<shafft::complexf> host(n, {0.0f, 0.0f});
  SHAFFT_CHECK(shafft::copyToBuffer(data, host.data(), n));
  SHAFFT_CHECK(fft.setBuffers(data, work));
  SHAFFT_CHECK(fft.execute(shafft::FFTDirection::FORWARD));

  std::vector<size_t> init_sub(3), init_off(3);
  std::vector<size_t> curr_sub(3), curr_off(3);
  SHAFFT_CHECK(fft.getLayout(init_sub, init_off, shafft::TensorLayout::INITIAL));
  SHAFFT_CHECK(fft.getLayout(curr_sub, curr_off, shafft::TensorLayout::CURRENT));

  bool match_initial = true;
  for (int i = 0; i < 3; ++i) {
    if (init_sub[i] != curr_sub[i] || init_off[i] != curr_off[i]) {
      match_initial = false;
    }
  }

  (void)shafft::freeBuffer(data);
  (void)shafft::freeBuffer(work);
  return match_initial;
}

// Test policy INITIAL supports backward execute after forward restored INITIAL layout.
static bool test_policy_initial_backward_from_restored_layout() {
  std::vector<size_t> dims = {32, 32, 32};
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  if (worldSize < 2)
    return true;

  std::vector<int> commDims(dims.size(), 1);
  commDims[0] = worldSize;

  shafft::FFTND fft;
  int rc = fft.init(
      commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD, shafft::TransformLayout::INITIAL);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  size_t n = fft.allocSize();
  shafft::complexf *data, *work;
  SHAFFT_CHECK(shafft::allocBuffer(n, &data));
  SHAFFT_CHECK(shafft::allocBuffer(n, &work));
  std::vector<shafft::complexf> host(n, {0.0f, 0.0f});
  SHAFFT_CHECK(shafft::copyToBuffer(data, host.data(), n));
  SHAFFT_CHECK(fft.setBuffers(data, work));

  SHAFFT_CHECK(fft.execute(shafft::FFTDirection::FORWARD));
  SHAFFT_CHECK(fft.execute(shafft::FFTDirection::BACKWARD));

  std::vector<size_t> init_sub(3), init_off(3);
  std::vector<size_t> curr_sub(3), curr_off(3);
  SHAFFT_CHECK(fft.getLayout(init_sub, init_off, shafft::TensorLayout::INITIAL));
  SHAFFT_CHECK(fft.getLayout(curr_sub, curr_off, shafft::TensorLayout::CURRENT));

  bool match_initial = true;
  for (int i = 0; i < 3; ++i) {
    if (init_sub[i] != curr_sub[i] || init_off[i] != curr_off[i]) {
      match_initial = false;
    }
  }

  (void)shafft::freeBuffer(data);
  (void)shafft::freeBuffer(work);
  return match_initial;
}

// Test policy REDISTRIBUTED keeps CURRENT layout as REDISTRIBUTED after forward execute
static bool test_policy_redistributed_keeps_layout() {
  std::vector<size_t> dims = {32, 32, 32};
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  if (worldSize < 2)
    return true;

  std::vector<int> commDims(dims.size(), 1);
  commDims[0] = worldSize;

  shafft::FFTND fft;
  int rc = fft.init(
      commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD, shafft::TransformLayout::REDISTRIBUTED);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  size_t n = fft.allocSize();
  shafft::complexf *data, *work;
  SHAFFT_CHECK(shafft::allocBuffer(n, &data));
  SHAFFT_CHECK(shafft::allocBuffer(n, &work));
  std::vector<shafft::complexf> host(n, {0.0f, 0.0f});
  SHAFFT_CHECK(shafft::copyToBuffer(data, host.data(), n));
  SHAFFT_CHECK(fft.setBuffers(data, work));
  SHAFFT_CHECK(fft.execute(shafft::FFTDirection::FORWARD));

  std::vector<size_t> trans_sub(3), trans_off(3);
  std::vector<size_t> curr_sub(3), curr_off(3);
  SHAFFT_CHECK(fft.getLayout(trans_sub, trans_off, shafft::TensorLayout::REDISTRIBUTED));
  SHAFFT_CHECK(fft.getLayout(curr_sub, curr_off, shafft::TensorLayout::CURRENT));

  bool match_redistributed = true;
  for (int i = 0; i < 3; ++i) {
    if (trans_sub[i] != curr_sub[i] || trans_off[i] != curr_off[i]) {
      match_redistributed = false;
    }
  }

  (void)shafft::freeBuffer(data);
  (void)shafft::freeBuffer(work);
  return match_redistributed;
}

// INITIAL Layout Tests

// Test: INITIAL layout distributes first nda axes
static bool test_initial_distributed_axes() {
  std::vector<size_t> dims = {64, 64, 32};
  const int ndim = static_cast<int>(dims.size());
  // nda = 1

  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  // Create commDims for nda=1
  std::vector<int> commDims(dims.size(), 1);
  commDims[0] = worldSize;

  shafft::FFTND fft;
  int rc = fft.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  std::vector<size_t> subsize(ndim), offset(ndim);
  SHAFFT_CHECK(fft.getLayout(subsize, offset, shafft::TensorLayout::INITIAL));

  if (worldSize > 1) {
    // First axis should be distributed (subsize < global) for nda=1
    for (int i = 0; i < 1; ++i) {
      if (subsize[i] >= dims[i])
        return false; // Should be less than global
    }
    // Remaining axes should be full (subsize == global)
    for (int i = 1; i < ndim; ++i) {
      if (subsize[i] != dims[i])
        return false;
    }
  } else {
    // Single rank: all axes full
    for (int i = 0; i < ndim; ++i) {
      if (subsize[i] != dims[i])
        return false;
    }
  }

  return true;
}

// Test: INITIAL layout with 2 distributed axes (nda=2)
static bool test_initial_nda2() {
  std::vector<size_t> dims = {16, 16, 16, 16}; // 4D tensor
  const int ndim = static_cast<int>(dims.size());
  // nda = 2

  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  if (worldSize < 4)
    return true; // Need at least 4 ranks for 2D decomposition

  // Create commDims for nda=2
  std::vector<int> commDims(dims.size(), 1);
  std::vector<int> mpi_dims(2, 0);
  MPI_Dims_create(worldSize, 2, mpi_dims.data());
  commDims[0] = mpi_dims[0];
  commDims[1] = mpi_dims[1];

  shafft::FFTND fft;
  int rc = fft.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  std::vector<size_t> subsize(ndim), offset(ndim);
  SHAFFT_CHECK(fft.getLayout(subsize, offset, shafft::TensorLayout::INITIAL));

  // First 2 axes should be distributed
  bool axis0_dist = (subsize[0] < dims[0]);
  bool axis1_dist = (subsize[1] < dims[1]);
  // Last 2 axes should be full
  bool axis2_full = (subsize[2] == dims[2]);
  bool axis3_full = (subsize[3] == dims[3]);

  return axis0_dist && axis1_dist && axis2_full && axis3_full;
}

// Test: INITIAL offset is zero for rank 0
static bool test_initial_rank0_offset() {
  std::vector<size_t> dims = {64, 64, 32};
  const int ndim = static_cast<int>(dims.size());
  // nda = 1

  int rank, worldSize;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  // Create commDims for nda=1
  std::vector<int> commDims(dims.size(), 1);
  commDims[0] = worldSize;

  shafft::FFTND fft;
  int rc = fft.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  rc = fft.plan();
  if (rc != 0)
    return false;

  std::vector<size_t> subsize(ndim), offset(ndim);
  SHAFFT_CHECK(fft.getLayout(subsize, offset, shafft::TensorLayout::INITIAL));

  if (rank == 0) {
    // Rank 0 should have offset 0 for all axes
    for (int i = 0; i < ndim; ++i) {
      if (offset[i] != 0)
        return false;
    }
  }

  // All ranks: non-distributed axes should have offset 0 (for nda=1)
  for (int i = 1; i < ndim; ++i) {
    if (offset[i] != 0)
      return false;
  }

  return true;
}

// Test: INITIAL layout with single rank (no distribution)
static bool test_initial_single_rank() {
  std::vector<size_t> dims = {64, 64, 32};
  const int ndim = static_cast<int>(dims.size());

  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  if (worldSize > 1)
    return true; // Only test with single rank

  // Create commDims for nda=1 (but single rank)
  std::vector<int> commDims(dims.size(), 1);
  commDims[0] = worldSize;

  shafft::FFTND fft;
  int rc = fft.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  std::vector<size_t> subsize(ndim), offset(ndim);
  SHAFFT_CHECK(fft.getLayout(subsize, offset, shafft::TensorLayout::INITIAL));

  // Single rank: subsize == dims, offset == 0
  for (int i = 0; i < ndim; ++i) {
    if (subsize[i] != dims[i])
      return false;
    if (offset[i] != 0)
      return false;
  }

  return true;
}

// Test: INITIAL layout is consistent across ranks (global view matches)
static bool test_initial_global_consistency() {
  std::vector<size_t> dims = {100, 50, 25}; // Non-power-of-2
  const int ndim = static_cast<int>(dims.size());
  // nda = 1

  int worldSize, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Create commDims for nda=1
  std::vector<int> commDims(dims.size(), 1);
  commDims[0] = worldSize;

  shafft::FFTND fft;
  int rc = fft.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  std::vector<size_t> subsize(ndim), offset(ndim);
  SHAFFT_CHECK(fft.getLayout(subsize, offset, shafft::TensorLayout::INITIAL));

  // Gather all offsets and subsizes for distributed axis
  std::vector<int> all_offsets(worldSize), allSizes(worldSize);
  MPI_Allgather(&offset[0], 1, MPI_INT, all_offsets.data(), 1, MPI_INT, MPI_COMM_WORLD);
  MPI_Allgather(&subsize[0], 1, MPI_INT, allSizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

  // Verify: offset[i] == sum of sizes before rank i
  int expected_offset = 0;
  for (int r = 0; r < worldSize; ++r) {
    if (all_offsets[r] != expected_offset)
      return false;
    expected_offset += allSizes[r];
  }

  // Total should equal global size
  if (static_cast<size_t>(expected_offset) != dims[0])
    return false;

  return true;
}

int main(int argc, char** argv) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::fprintf(stderr, "MPI_Init failed\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  test::TestRunnerMPI runner("Layout Tests");

  runner.run("REDISTRIBUTED differs from INITIAL", test_redistributed_layout);
  runner.run("CURRENT tracks execute() state", test_current_tracks_state);
  runner.run("Policy INITIAL restores CURRENT layout", test_policy_initial_restores_layout);
  runner.run("Policy INITIAL supports backward after restore",
             test_policy_initial_backward_from_restored_layout);
  runner.run("Policy REDISTRIBUTED keeps CURRENT layout", test_policy_redistributed_keeps_layout);

  // INITIAL layout tests
  runner.run("INITIAL distributes first nda axes", test_initial_distributed_axes);
  runner.run("INITIAL with nda=2 (4D tensor)", test_initial_nda2);
  runner.run("INITIAL rank 0 offset is zero", test_initial_rank0_offset);
  runner.run("INITIAL single rank (no distribution)", test_initial_single_rank);
  runner.run("INITIAL global consistency", test_initial_global_consistency);

  int result = runner.finalize();
  MPI_Finalize();
  return result;
}
