/**
 * @file test_inactive_ranks.cpp
 * @brief Test graceful handling of inactive (excluded) MPI ranks
 *
 * Tests that SHAFFT handles inactive ranks gracefully when the tensor
 * decomposition doesn't use all available MPI ranks.
 *
 * Run with: mpirun -np 5 ./test_inactive_ranks
 * (5 ranks on a 4x4x4 tensor with nda=1 will have 1 inactive rank)
 */
#include "test_utils.hpp"
#include <cstring>
#include <mpi.h>
#include <shafft/shafft.hpp>
#include <vector>

// RAII wrapper for portable device/host buffers
struct DeviceBuffers {
  shafft::complexf* data = nullptr;
  shafft::complexf* work = nullptr;
  ~DeviceBuffers() {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
  }
};

// Copy host vectors into SHAFFT-managed buffers and attach to plan
static bool set_plan_buffers(shafft::FFTND& fft,
                             const std::vector<shafft::complexf>& host_data,
                             DeviceBuffers& bufs) {
  const size_t localAllocSize = fft.allocSize();
  if (!fft.isActive())
    return true; // Nothing to do for inactive ranks

  if (shafft::allocBuffer(localAllocSize, &bufs.data) != 0)
    return false;
  if (shafft::allocBuffer(localAllocSize, &bufs.work) != 0)
    return false;

  if (shafft::copyToBuffer(bufs.data, host_data.data(), localAllocSize) != 0)
    return false;

  int rc = fft.setBuffers(bufs.data, bufs.work);
  return rc == 0;
}

// Test: Plan creation succeeds on all ranks (active and inactive)
static bool test_plan_create_with_inactive() {
  // Use a small tensor with nda=1. With 5 ranks, if tensor size[0]=4,
  // only 4 ranks will be active, and 1 will be inactive.
  std::vector<size_t> dims = {4, 4, 4};
  std::vector<int> commDims = test::compute_comm_dims_nda1(dims, MPI_COMM_WORLD);

  shafft::FFTND fft;
  int rc = fft.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();

  // Plan creation should succeed on ALL ranks (active and inactive)
  return rc == 0;
}

// Test: isActive() correctly identifies active/inactive ranks
static bool test_is_active_query() {
  std::vector<size_t> dims = {4, 4, 4};
  std::vector<int> commDims = test::compute_comm_dims_nda1(dims, MPI_COMM_WORLD);

  shafft::FFTND fft;
  int rc = fft.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  bool is_active = fft.isActive();

  // With dims[0]=4 and nda=1, we can have at most 4 active ranks
  // Ranks 0-3 should be active, rank 4+ should be inactive
  if (size <= 4) {
    // All ranks should be active
    return is_active == true;
  } else {
    // Ranks 0 to 3 active, ranks 4+ inactive
    bool expected_active = (rank < 4);
    return is_active == expected_active;
  }
}

// Test: allocSize() returns 0 for inactive ranks
static bool test_alloc_size_inactive() {
  std::vector<size_t> dims = {4, 4, 4};
  std::vector<int> commDims = test::compute_comm_dims_nda1(dims, MPI_COMM_WORLD);

  shafft::FFTND fft;
  int rc = fft.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  size_t localAllocSize = fft.allocSize();

  if (!fft.isActive()) {
    // Inactive ranks should have allocSize() == 0
    return localAllocSize == 0;
  } else {
    // Active ranks should have non-zero allocSize
    return localAllocSize > 0;
  }
}

// Test: execute() succeeds (no-op) on inactive ranks
static bool test_execute_inactive() {
  std::vector<size_t> dims = {4, 4, 4};
  std::vector<int> commDims = test::compute_comm_dims_nda1(dims, MPI_COMM_WORLD);

  shafft::FFTND fft;
  int rc = fft.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  size_t localAllocSize = fft.allocSize();

  // Allocate buffers only for active ranks
  DeviceBuffers bufs;
  if (fft.isActive()) {
    std::vector<shafft::complexf> host(localAllocSize);
    for (size_t i = 0; i < localAllocSize; ++i) {
      host[i] = shafft::complexf(static_cast<float>(i), 0.0f);
    }
    if (!set_plan_buffers(fft, host, bufs))
      return false;
  }

  // Execute should succeed on all ranks
  rc = fft.execute(shafft::FFTDirection::FORWARD);
  if (rc != 0)
    return false;

  // Backward should also succeed
  rc = fft.execute(shafft::FFTDirection::BACKWARD);
  if (rc != 0)
    return false;

  return true;
}

// Test: normalize() succeeds (no-op) on inactive ranks
static bool test_normalize_inactive() {
  std::vector<size_t> dims = {4, 4, 4};
  std::vector<int> commDims = test::compute_comm_dims_nda1(dims, MPI_COMM_WORLD);

  shafft::FFTND fft;
  int rc = fft.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  size_t localAllocSize = fft.allocSize();

  DeviceBuffers bufs;
  if (fft.isActive()) {
    std::vector<shafft::complexf> host(localAllocSize, shafft::complexf(1.0f, 0.0f));
    if (!set_plan_buffers(fft, host, bufs))
      return false;
  }

  // Execute forward
  rc = fft.execute(shafft::FFTDirection::FORWARD);
  if (rc != 0)
    return false;

  // Normalize should succeed on all ranks
  rc = fft.normalize();
  if (rc != 0)
    return false;

  return true;
}

// Test: getLayout() returns zeros for inactive ranks
static bool test_get_layout_inactive() {
  std::vector<size_t> dims = {4, 4, 4};
  std::vector<int> commDims = test::compute_comm_dims_nda1(dims, MPI_COMM_WORLD);

  shafft::FFTND fft;
  int rc = fft.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  std::vector<size_t> subsize(3), offset(3);
  rc = fft.getLayout(subsize, offset);
  if (rc != 0)
    return false;

  if (!fft.isActive()) {
    // Inactive ranks should have all-zero layout
    for (int i = 0; i < 3; ++i) {
      if (subsize[i] != 0 || offset[i] != 0)
        return false;
    }
  } else {
    // Active ranks should have positive subsize
    for (int i = 0; i < 3; ++i) {
      if (subsize[i] <= 0)
        return false;
    }
  }

  return true;
}

// Test: Full roundtrip works correctly with inactive ranks present
static bool test_roundtrip_with_inactive() {
  std::vector<size_t> dims = {4, 4, 4};
  std::vector<int> commDims = test::compute_comm_dims_nda1(dims, MPI_COMM_WORLD);

  shafft::FFTND fft;
  int rc = fft.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  size_t localAllocSize = fft.allocSize();

  DeviceBuffers bufs;
  std::vector<shafft::complexf> original;
  if (fft.isActive()) {
    original.resize(localAllocSize);
    for (size_t i = 0; i < localAllocSize; ++i) {
      original[i] = shafft::complexf(static_cast<float>(i % 7), static_cast<float>(i % 3));
    }
    if (!set_plan_buffers(fft, original, bufs))
      return false;
  }

  // Forward
  rc = fft.execute(shafft::FFTDirection::FORWARD);
  if (rc != 0)
    return false;

  // Backward
  rc = fft.execute(shafft::FFTDirection::BACKWARD);
  if (rc != 0)
    return false;

  // Normalize
  rc = fft.normalize();
  if (rc != 0)
    return false;

  // Verify roundtrip for active ranks
  if (fft.isActive()) {
    shafft::complexf* result_dev = nullptr;
    shafft::complexf* work_dev = nullptr;
    rc = fft.getBuffers(&result_dev, &work_dev);
    if (rc != 0)
      return false;

    std::vector<shafft::complexf> result(localAllocSize);
    if (shafft::copyFromBuffer(result.data(), result_dev, localAllocSize) != 0)
      return false;
    for (size_t i = 0; i < localAllocSize; ++i) {
      if (!test::approx_eq(result[i], original[i], 1e-4f)) {
        return false;
      }
    }
  }

  return true;
}

// Test: Plan release works correctly on inactive ranks
static bool test_release_inactive() {
  std::vector<size_t> dims = {4, 4, 4};
  std::vector<int> commDims = test::compute_comm_dims_nda1(dims, MPI_COMM_WORLD);

  shafft::FFTND fft;
  int rc = fft.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  // Release should work on all ranks
  fft.release();

  // After release, plan should not be configured
  return !fft.isConfigured();
}

// Main
int main(int argc, char** argv) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::fprintf(stderr, "MPI_Init failed\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  test::TestRunnerMPI runner("Inactive Ranks Tests");

  runner.run("Plan creation with inactive ranks", test_plan_create_with_inactive);
  runner.run("isActive() query", test_is_active_query);
  runner.run("allocSize() for inactive ranks", test_alloc_size_inactive);
  runner.run("execute() on inactive ranks", test_execute_inactive);
  runner.run("normalize() on inactive ranks", test_normalize_inactive);
  runner.run("getLayout() for inactive ranks", test_get_layout_inactive);
  runner.run("Roundtrip with inactive ranks", test_roundtrip_with_inactive);
  runner.run("release() on inactive ranks", test_release_inactive);

  int ret = runner.finalize();

  MPI_Finalize();
  return ret;
}
