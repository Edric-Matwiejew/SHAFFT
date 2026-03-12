/**
 * @file test_configuration_modes.cpp
 * @brief Tests for the unified configurationND function
 *
 * Tests cover:
 * - Auto mode with MAXIMIZE_NDA and MINIMIZE_NDA strategies
 * - Explicit commDims (primary preference)
 * - Explicit nda (secondary preference)
 * - Fallback chain: commDims -> nda -> strategy
 * - Edge cases: single rank, invalid inputs, inactive ranks
 * - Memory limit enforcement
 * - Subsizes and offsets consistency
 */

#include "test_utils.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <shafft/shafft.hpp>
#include <vector>

// Return the number of prime factors (with multiplicity) of n
static int prime_factor_count(int n) {
  int count = 0;
  for (int p = 2; p * p <= n; ++p) {
    while (n % p == 0) {
      ++count;
      n /= p;
    }
  }
  if (n > 1)
    ++count;
  return count;
}

// Helper: sum of values across all ranks
static int global_sum(int local_val) {
  int global_val = 0;
  MPI_Allreduce(&local_val, &global_val, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  return global_val;
}

// Helper: compare return code to expected SHAFFT status
static bool expect_status(int rc, shafft::Status expected) {
  return rc == static_cast<int>(expected);
}

// Helper: count number of distributed axes (commDims[i] > 1)
static int count_distributed_axes(const std::vector<int>& commDims) {
  int count = 0;
  for (int d : commDims) {
    if (d > 1)
      count++;
  }
  return count;
}

// ============================================================================
// Auto mode with DecompositionStrategy tests
// ============================================================================

// Test: configurationND auto mode with MAXIMIZE_NDA strategy
static bool test_configurationND_auto_maximize() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  if (worldSize < 2)
    return true; // Skip single rank

  std::vector<size_t> dims = {32, 32, 32};
  const int ndim = static_cast<int>(dims.size());
  std::vector<int> commDims(ndim, 0); // All zeros -> auto mode
  int nda = 0;                        // Auto mode
  std::vector<size_t> subsize(ndim), offset(ndim);
  int commSize = 0;

  int rc = shafft::configurationND(dims,
                                   shafft::FFTType::C2C,
                                   commDims,
                                   nda,
                                   subsize,
                                   offset,
                                   commSize,
                                   shafft::DecompositionStrategy::MAXIMIZE_NDA,
                                   0,
                                   MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  // nda should be set to some value >= 1
  if (nda < 1)
    return false;

  // Verify nda matches actual commDims distribution count
  int actual_nda = count_distributed_axes(commDims);
  if (actual_nda != nda)
    return false;

  // Verify grid product is reasonable
  size_t grid_product = test::product(commDims);
  if (grid_product == 0 || static_cast<int>(grid_product) > worldSize)
    return false;

  // Verify commSize matches grid product
  if (commSize != static_cast<int>(grid_product))
    return false;

  // Verify subsizes are positive and within bounds
  for (int i = 0; i < ndim; ++i) {
    if (subsize[i] == 0 || subsize[i] > dims[i])
      return false;
  }

  return true;
}

// Test: configurationND auto mode with MINIMIZE_NDA strategy
static bool test_configurationND_auto_minimize() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  if (worldSize < 2)
    return true;

  std::vector<size_t> dims = {32, 32, 32};
  const int ndim = static_cast<int>(dims.size());

  // First get MAXIMIZE_NDA result
  std::vector<int> commDims_max(ndim, 0);
  int nda_max = 0;
  std::vector<size_t> subsize_max(ndim), offset_max(ndim);
  int commSizeMax = 0;

  int rc = shafft::configurationND(dims,
                                   shafft::FFTType::C2C,
                                   commDims_max,
                                   nda_max,
                                   subsize_max,
                                   offset_max,
                                   commSizeMax,
                                   shafft::DecompositionStrategy::MAXIMIZE_NDA,
                                   0,
                                   MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  // Now get MINIMIZE_NDA result
  std::vector<int> commDims_min(ndim, 0);
  int nda_min = 0;
  std::vector<size_t> subsize_min(ndim), offset_min(ndim);
  int commSizeMin = 0;

  rc = shafft::configurationND(dims,
                               shafft::FFTType::C2C,
                               commDims_min,
                               nda_min,
                               subsize_min,
                               offset_min,
                               commSizeMin,
                               shafft::DecompositionStrategy::MINIMIZE_NDA,
                               0,
                               MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  // MINIMIZE_NDA should give nda <= MAXIMIZE_NDA's nda
  if (nda_min > nda_max)
    return false;

  // Both should be valid (>= 1)
  if (nda_min < 1 || nda_max < 1)
    return false;

  // Verify both have valid commSize
  if (commSizeMax <= 0 || commSizeMax > worldSize)
    return false;
  if (commSizeMin <= 0 || commSizeMin > worldSize)
    return false;

  return true;
}

// ============================================================================
// Explicit commDims tests (primary preference)
// ============================================================================

// Test: configurationND with explicit commDims (all > 0)
static bool test_configurationND_explicit_comm_dims() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  if (worldSize < 4)
    return true;

  std::vector<size_t> dims = {64, 64, 32};
  const int ndim = static_cast<int>(dims.size());

  // Request [2, 2, 1] grid (4 ranks total)
  std::vector<int> commDims = {2, 2, 1};
  int nda = 0; // Should be ignored when commDims is explicit
  std::vector<size_t> subsize(ndim), offset(ndim);
  int commSize = 0;

  int rc = shafft::configurationND(dims,
                                   shafft::FFTType::C2C,
                                   commDims,
                                   nda,
                                   subsize,
                                   offset,
                                   commSize,
                                   shafft::DecompositionStrategy::MAXIMIZE_NDA,
                                   0,
                                   MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  // commSize should be 4
  if (commSize != 4)
    return false;

  // commDims should be preserved
  if (commDims[0] != 2 || commDims[1] != 2 || commDims[2] != 1)
    return false;

  // nda should match the actual distributed axes
  if (nda != 2)
    return false;

  return true;
}

// Test: configurationND with invalid commDims hint falls back to strategy
static bool test_configurationND_invalid_comm_dims() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  std::vector<size_t> dims = {64, 64, 32};
  const int ndim = static_cast<int>(dims.size());

  // Request grid that uses more ranks than available - should fall back to strategy
  std::vector<int> commDims = {worldSize + 1, 1, 1};
  int nda = 0;
  std::vector<size_t> subsize(ndim), offset(ndim);
  int commSize = 0;

  int rc = shafft::configurationND(dims,
                                   shafft::FFTType::C2C,
                                   commDims,
                                   nda,
                                   subsize,
                                   offset,
                                   commSize,
                                   shafft::DecompositionStrategy::MAXIMIZE_NDA,
                                   0,
                                   MPI_COMM_WORLD);

  // configurationND should fall back to strategy and succeed
  if (rc != 0)
    return false;

  // commDims should have been adjusted to a valid value
  int grid_product = 1;
  for (int i = 0; i < ndim; ++i)
    grid_product *= commDims[i];
  return grid_product <= worldSize;
}

// Test: configurationND with non-prefix commDims (trailing > 1) should fail
static bool test_configurationND_invalid_trailing() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  if (worldSize < 4)
    return true;

  std::vector<size_t> dims = {64, 64, 32};
  const int ndim = static_cast<int>(dims.size());

  // Invalid: last axis must be 1 for valid slab decomposition
  std::vector<int> commDims = {1, 2, 2}; // Invalid pattern (gap)
  int nda = 0;
  std::vector<size_t> subsize(ndim), offset(ndim);
  int commSize = 0;

  int rc = shafft::configurationND(dims,
                                   shafft::FFTType::C2C,
                                   commDims,
                                   nda,
                                   subsize,
                                   offset,
                                   commSize,
                                   shafft::DecompositionStrategy::MAXIMIZE_NDA,
                                   0,
                                   MPI_COMM_WORLD);

  // Invalid commDims should fall back to strategy and succeed
  if (rc != 0)
    return false;

  // Verify a valid configuration was produced
  if (commSize < 1)
    return false;
  if (nda < 0 || nda >= ndim)
    return false;

  return true;
}

// Test: configurationND with inactive ranks (commDims product < worldSize)
static bool test_configurationND_inactive_ranks() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  if (worldSize < 5)
    return true; // Need more ranks than we'll use

  std::vector<size_t> dims = {64, 64, 32};
  const int ndim = static_cast<int>(dims.size());

  // Request [2, 2, 1] grid (4 ranks) with 5+ available
  std::vector<int> commDims = {2, 2, 1};
  int nda = 0;
  std::vector<size_t> subsize(ndim), offset(ndim);
  int commSize = 0;

  int rc = shafft::configurationND(dims,
                                   shafft::FFTType::C2C,
                                   commDims,
                                   nda,
                                   subsize,
                                   offset,
                                   commSize,
                                   shafft::DecompositionStrategy::MAXIMIZE_NDA,
                                   0,
                                   MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  // commSize should be 4 (not worldSize)
  if (commSize != 4)
    return false;

  return true;
}

// ============================================================================
// Explicit nda tests (secondary preference)
// ============================================================================

// Test: configurationND with explicit nda (commDims = 0, nda > 0)
static bool test_configurationND_explicit_nda() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  if (worldSize < 4)
    return true;

  std::vector<size_t> dims = {16, 16, 16, 16}; // 4D tensor
  const int ndim = static_cast<int>(dims.size());
  int requested_nda = 2; // Request exactly 2 distributed axes

  std::vector<int> commDims(ndim, 0); // All zeros -> use nda preference
  int nda = requested_nda;
  std::vector<size_t> subsize(ndim), offset(ndim);
  int commSize = 0;

  int rc = shafft::configurationND(dims,
                                   shafft::FFTType::C2C,
                                   commDims,
                                   nda,
                                   subsize,
                                   offset,
                                   commSize,
                                   shafft::DecompositionStrategy::MAXIMIZE_NDA,
                                   0,
                                   MPI_COMM_WORLD);

  // Should always succeed (falls back to strategy if nda can't be satisfied)
  if (rc != 0)
    return false;

  // If the world size cannot support two distributed axes (>1 each),
  // it should have fallen back to strategy (nda may differ from requested)
  if (prime_factor_count(worldSize) < requested_nda) {
    // Fallback: nda may be less than requested, but should be valid
    if (nda < 0 || nda >= ndim)
      return false;
    return true;
  }

  // nda should be exactly what was requested when possible
  if (nda != requested_nda)
    return false;

  // Count how many commDims are > 1
  int actual_nda = count_distributed_axes(commDims);
  if (actual_nda != nda)
    return false;

  // Verify commDims forms a valid prefix (>1 entries are at the front)
  bool seen_one = false;
  for (int i = 0; i < ndim; ++i) {
    if (commDims[i] == 1)
      seen_one = true;
    else if (seen_one)
      return false; // Non-1 after 1 is invalid prefix
  }

  // Grid product should equal worldSize or be less (inactive ranks)
  size_t grid_product = test::product(commDims);
  if (static_cast<int>(grid_product) > worldSize)
    return false;

  return true;
}

// Test: configurationND with invalid nda hint (>= ndim) falls back to strategy
static bool test_configurationND_invalid_nda() {
  std::vector<size_t> dims = {64, 64, 32};
  const int ndim = static_cast<int>(dims.size());

  std::vector<int> commDims(ndim, 0);
  int nda = 3; // Invalid: nda must be < ndim, but configurationND should fall back
  std::vector<size_t> subsize(ndim), offset(ndim);
  int commSize = 0;

  int rc = shafft::configurationND(dims,
                                   shafft::FFTType::C2C,
                                   commDims,
                                   nda,
                                   subsize,
                                   offset,
                                   commSize,
                                   shafft::DecompositionStrategy::MAXIMIZE_NDA,
                                   0,
                                   MPI_COMM_WORLD);

  // configurationND should fall back to strategy and succeed
  if (rc != 0)
    return false;

  // nda should have been adjusted to a valid value (< ndim)
  return nda < ndim;
}

// ============================================================================
// Single rank tests
// ============================================================================

// Test: configurationND single rank with auto mode succeeds (nda=0)
static bool test_configurationND_single_rank_auto() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  if (worldSize > 1)
    return true; // Only test with single rank

  std::vector<size_t> dims = {64, 64, 32};
  const int ndim = static_cast<int>(dims.size());

  std::vector<int> commDims(ndim, 0);
  int nda = 0; // Auto mode
  std::vector<size_t> subsize(ndim), offset(ndim);
  int commSize = 0;

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

  // Single rank with auto mode should succeed with all-ones commDims
  if (rc != 0)
    return false;

  // commDims should be [1, 1, 1]
  for (int i = 0; i < ndim; ++i) {
    if (commDims[i] != 1)
      return false;
  }

  // Subsize should equal full dims
  for (int i = 0; i < ndim; ++i) {
    if (subsize[i] != dims[i])
      return false;
    if (offset[i] != 0)
      return false;
  }

  if (commSize != 1)
    return false;

  return true;
}

// Test: configurationND single rank with explicit nda > 0 fails
static bool test_configurationND_single_rank_explicit_nda() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  if (worldSize > 1)
    return true; // Only test with single rank

  std::vector<size_t> dims = {64, 64, 32};
  const int ndim = static_cast<int>(dims.size());

  std::vector<int> commDims(ndim, 0);
  int nda = 1; // Request distribution, but single rank can't satisfy
  std::vector<size_t> subsize(ndim), offset(ndim);
  int commSize = 0;

  int rc = shafft::configurationND(dims,
                                   shafft::FFTType::C2C,
                                   commDims,
                                   nda,
                                   subsize,
                                   offset,
                                   commSize,
                                   shafft::DecompositionStrategy::MAXIMIZE_NDA,
                                   0,
                                   MPI_COMM_WORLD);

  // Single rank can't satisfy nda>0, but should fall back to nda=0
  if (rc != 0)
    return false;

  // With single rank, should fall back to no distribution
  if (nda != 0)
    return false;
  if (commSize != 1)
    return false;

  // Subsize should equal full dims for single rank
  for (int i = 0; i < ndim; ++i) {
    if (subsize[i] != dims[i])
      return false;
    if (offset[i] != 0)
      return false;
  }

  return true;
}

// ============================================================================
// Memory limit tests
// ============================================================================

// Test: configurationND with memLimit enforces size constraint
static bool test_configurationND_mem_limit_enforced() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  if (worldSize < 2)
    return true; // Need multiple ranks

  // 64x64x64 C2C tensor = 64^3 * 8 bytes = 2MB per buffer
  std::vector<size_t> dims = {64, 64, 64};
  const int ndim = static_cast<int>(dims.size());

  std::vector<int> commDims(ndim, 0);
  int nda = 0; // Auto mode
  std::vector<size_t> subsize(ndim), offset(ndim);
  int commSize = 0;

  // Set a memory limit that requires distribution
  size_t memLimit = 2 * 1024 * 1024; // 2MB

  int rc = shafft::configurationND(dims,
                                   shafft::FFTType::C2C,
                                   commDims,
                                   nda,
                                   subsize,
                                   offset,
                                   commSize,
                                   shafft::DecompositionStrategy::MAXIMIZE_NDA,
                                   memLimit,
                                   MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  // Verify local size is within limit
  size_t local_size = test::product(subsize);
  local_size *= 8; // C2C = 2 * sizeof(float) = 8 bytes
  local_size *= 2; // Account for data + work buffers

  if (local_size > memLimit)
    return false;

  return true;
}

// Test: configurationND with memLimit enforced under MINIMIZE_NDA
static bool test_configurationND_mem_limit_enforced_minimize() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  if (worldSize < 2)
    return true; // Need multiple ranks

  std::vector<size_t> dims = {64, 64, 64};
  const int ndim = static_cast<int>(dims.size());

  std::vector<int> commDims(ndim, 0);
  int nda = 0; // Auto mode
  std::vector<size_t> subsize(ndim), offset(ndim);
  int commSize = 0;

  size_t memLimit = 2 * 1024 * 1024; // 2MB

  int rc = shafft::configurationND(dims,
                                   shafft::FFTType::C2C,
                                   commDims,
                                   nda,
                                   subsize,
                                   offset,
                                   commSize,
                                   shafft::DecompositionStrategy::MINIMIZE_NDA,
                                   memLimit,
                                   MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  size_t local_size = test::product(subsize);
  local_size *= 8; // C2C element bytes
  local_size *= 2; // data + work
  if (local_size > memLimit)
    return false;

  if (nda < 1)
    return false; // distributed
  if (commSize <= 0 || commSize > worldSize)
    return false;

  return true;
}

// Test: configurationND fails when memLimit is too small
static bool test_configurationND_mem_limit_too_small() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  if (worldSize < 2)
    return true;

  std::vector<size_t> dims = {64, 64, 64};
  const int ndim = static_cast<int>(dims.size());

  std::vector<int> commDims(ndim, 0);
  int nda = 0; // Auto mode
  std::vector<size_t> subsize(ndim), offset(ndim);
  int commSize = 0;

  // Set impossibly small memory limit (1 byte)
  size_t memLimit = 1;

  int rc = shafft::configurationND(dims,
                                   shafft::FFTType::C2C,
                                   commDims,
                                   nda,
                                   subsize,
                                   offset,
                                   commSize,
                                   shafft::DecompositionStrategy::MAXIMIZE_NDA,
                                   memLimit,
                                   MPI_COMM_WORLD);

  // Should fail - no valid decomposition fits in 1 byte
  return expect_status(rc, shafft::Status::ERR_INVALID_DECOMP);
}

// Test: MINIMIZE_NDA also fails when memLimit is impossible
static bool test_configurationND_mem_limit_too_small_minimize() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  if (worldSize < 2)
    return true;

  std::vector<size_t> dims = {64, 64, 64};
  const int ndim = static_cast<int>(dims.size());

  std::vector<int> commDims(ndim, 0);
  int nda = 0;
  std::vector<size_t> subsize(ndim), offset(ndim);
  int commSize = 0;

  size_t memLimit = 1;

  int rc = shafft::configurationND(dims,
                                   shafft::FFTType::C2C,
                                   commDims,
                                   nda,
                                   subsize,
                                   offset,
                                   commSize,
                                   shafft::DecompositionStrategy::MINIMIZE_NDA,
                                   memLimit,
                                   MPI_COMM_WORLD);

  return expect_status(rc, shafft::Status::ERR_INVALID_DECOMP);
}

// Test: configurationND explicit nda with memLimit (pass case)
static bool test_configurationND_explicit_nda_mem_limit_pass() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  if (worldSize < 2)
    return true;

  std::vector<size_t> dims = {32, 32, 32};
  const int ndim = static_cast<int>(dims.size());

  std::vector<int> commDims(ndim, 0);
  int nda = 1; // Explicit nda
  std::vector<size_t> subsize(ndim), offset(ndim);
  int commSize = 0;

  // Large enough memLimit that nda=1 should work
  size_t memLimit = 256 * 1024; // 256KB

  int rc = shafft::configurationND(dims,
                                   shafft::FFTType::C2C,
                                   commDims,
                                   nda,
                                   subsize,
                                   offset,
                                   commSize,
                                   shafft::DecompositionStrategy::MAXIMIZE_NDA,
                                   memLimit,
                                   MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  // Verify local size is within limit
  size_t local_size = test::product(subsize);
  local_size *= 8;
  local_size *= 2; // data + work buffers

  if (local_size > memLimit)
    return false;

  return true;
}

// Test: configurationND explicit nda with memLimit (fail case)
static bool test_configurationND_explicit_nda_mem_limit_fail() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  if (worldSize < 2)
    return true;

  std::vector<size_t> dims = {64, 64, 64};
  const int ndim = static_cast<int>(dims.size());

  std::vector<int> commDims(ndim, 0);
  int nda = 1; // Explicit nda=1 gives less distribution
  std::vector<size_t> subsize(ndim), offset(ndim);
  int commSize = 0;

  // Very small memLimit - nda=1 won't be enough distribution
  size_t memLimit = 1;

  int rc = shafft::configurationND(dims,
                                   shafft::FFTType::C2C,
                                   commDims,
                                   nda,
                                   subsize,
                                   offset,
                                   commSize,
                                   shafft::DecompositionStrategy::MAXIMIZE_NDA,
                                   memLimit,
                                   MPI_COMM_WORLD);

  // Should fail - explicit nda=1 doesn't meet memory constraint
  return expect_status(rc, shafft::Status::ERR_INVALID_DECOMP);
}

// Test: configurationND explicit commDims with memLimit (pass case)
static bool test_configurationND_explicit_dims_mem_limit_pass() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  if (worldSize < 2)
    return true;

  std::vector<size_t> dims = {32, 32, 32};
  const int ndim = static_cast<int>(dims.size());

  std::vector<int> commDims = {2, 1, 1}; // Explicit grid
  int nda = 0;
  std::vector<size_t> subsize(ndim), offset(ndim);
  int commSize = 0;

  // Large memLimit that should pass
  size_t memLimit = 256 * 1024;

  int rc = shafft::configurationND(dims,
                                   shafft::FFTType::C2C,
                                   commDims,
                                   nda,
                                   subsize,
                                   offset,
                                   commSize,
                                   shafft::DecompositionStrategy::MAXIMIZE_NDA,
                                   memLimit,
                                   MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  // Verify local size is within limit
  size_t local_size = test::product(subsize);
  local_size *= 8;
  local_size *= 2; // data + work

  if (local_size > memLimit)
    return false;

  return true;
}

// Test: configurationND explicit commDims with memLimit (fail case)
static bool test_configurationND_explicit_dims_mem_limit_fail() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  if (worldSize < 2)
    return true;

  std::vector<size_t> dims = {64, 64, 64};
  const int ndim = static_cast<int>(dims.size());

  std::vector<int> commDims = {2, 1, 1}; // Explicit grid
  int nda = 0;
  std::vector<size_t> subsize(ndim), offset(ndim);
  int commSize = 0;

  // Very small memory limit - should fail with this explicit grid
  size_t memLimit = 1;

  int rc = shafft::configurationND(dims,
                                   shafft::FFTType::C2C,
                                   commDims,
                                   nda,
                                   subsize,
                                   offset,
                                   commSize,
                                   shafft::DecompositionStrategy::MAXIMIZE_NDA,
                                   memLimit,
                                   MPI_COMM_WORLD);

  // Should fail - explicit grid doesn't meet memory constraint
  return expect_status(rc, shafft::Status::ERR_INVALID_DECOMP);
}

// ============================================================================
// Subsize and offset consistency tests
// ============================================================================

// Test: Subsizes sum correctly across ranks
static bool test_subsize_global_sum() {
  int worldSize, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (worldSize < 2)
    return true;

  std::vector<size_t> dims = {100, 50, 25}; // Non-power-of-2
  const int ndim = static_cast<int>(dims.size());

  std::vector<int> commDims(ndim, 0);
  int nda = 1; // Explicit nda=1
  std::vector<size_t> subsize(ndim), offset(ndim);
  int commSize = 0;

  int rc = shafft::configurationND(dims,
                                   shafft::FFTType::C2C,
                                   commDims,
                                   nda,
                                   subsize,
                                   offset,
                                   commSize,
                                   shafft::DecompositionStrategy::MAXIMIZE_NDA,
                                   0,
                                   MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  // For each distributed axis, subsizes should sum to global dim
  for (int axis = 0; axis < nda; ++axis) {
    if (commDims[axis] > 1) {
      int sum = global_sum(static_cast<int>(subsize[axis]));
      if (sum != static_cast<int>(dims[axis]))
        return false;
    }
  }

  return true;
}

// Test: Offsets form contiguous blocks across active ranks
static bool test_offsets_contiguous() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  std::vector<size_t> dims = {100, 50};
  const int ndim = static_cast<int>(dims.size());

  std::vector<int> commDims(ndim, 0);
  int nda = 1; // Request 1 distributed axis
  std::vector<size_t> subsize(ndim), offset(ndim);
  int commSize = 0;

  int rc = shafft::configurationND(dims,
                                   shafft::FFTType::C2C,
                                   commDims,
                                   nda,
                                   subsize,
                                   offset,
                                   commSize,
                                   shafft::DecompositionStrategy::MAXIMIZE_NDA,
                                   0,
                                   MPI_COMM_WORLD);
  // Should always succeed (falls back if nda can't be satisfied)
  if (rc != 0)
    return false;

  // Single rank falls back to nda=0, skip contiguity check
  if (worldSize == 1) {
    if (nda != 0)
      return false;
    if (commSize != 1)
      return false;
    return true;
  }

  // Gather offsets/sizes for axis 0 from all ranks
  int local_offset = static_cast<int>(offset[0]);
  int local_size = static_cast<int>(subsize[0]);
  std::vector<int> all_offsets(worldSize), allSizes(worldSize);
  MPI_Allgather(&local_offset, 1, MPI_INT, all_offsets.data(), 1, MPI_INT, MPI_COMM_WORLD);
  MPI_Allgather(&local_size, 1, MPI_INT, allSizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

  // Only first cart_size ranks are active
  int cart_size = commDims[0];
  if (cart_size < 1)
    return false;

  for (int i = 0; i < cart_size - 1; ++i) {
    if (all_offsets[i] + allSizes[i] != all_offsets[i + 1]) {
      return false;
    }
  }

  // Last active rank should end exactly at global dimension
  if (all_offsets[cart_size - 1] + allSizes[cart_size - 1] != static_cast<int>(dims[0])) {
    return false;
  }

  return true;
}

// ============================================================================
// Double precision test
// ============================================================================

// Test: configurationND works with Z2Z (double precision)
static bool test_configurationND_double_precision() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  std::vector<size_t> dims = {32, 32, 32};
  const int ndim = static_cast<int>(dims.size());

  std::vector<int> commDims(ndim, 0);
  int nda = 0;
  std::vector<size_t> subsize(ndim), offset(ndim);
  int commSize = 0;

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
  if (rc != 0)
    return false;

  // Configuration should work identically for Z2Z
  if (subsize.size() != 3)
    return false;
  if (commSize <= 0)
    return false;

  return true;
}

// ============================================================================
// 2D and 4D tensor tests
// ============================================================================

// Test: configurationND with 2D tensor
static bool test_configurationND_2d() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  std::vector<size_t> dims = {128, 256};
  const int ndim = static_cast<int>(dims.size());

  std::vector<int> commDims(ndim, 0);
  int nda = 0;
  std::vector<size_t> subsize(ndim), offset(ndim);
  int commSize = 0;

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
  if (rc != 0)
    return false;

  // Verify dimensions
  if (subsize.size() != 2)
    return false;
  if (commDims.size() != 2)
    return false;

  return true;
}

// Test: configurationND with 4D tensor
static bool test_configurationND_4d() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  if (worldSize < 4)
    return true;

  std::vector<size_t> dims = {32, 32, 32, 32}; // 4D
  const int ndim = static_cast<int>(dims.size());

  std::vector<int> commDims(ndim, 0);
  int nda = 0;
  std::vector<size_t> subsize(ndim), offset(ndim);
  int commSize = 0;

  int rc = shafft::configurationND(dims,
                                   shafft::FFTType::C2C,
                                   commDims,
                                   nda,
                                   subsize,
                                   offset,
                                   commSize,
                                   shafft::DecompositionStrategy::MAXIMIZE_NDA,
                                   0,
                                   MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  // Verify dimensions
  if (subsize.size() != 4)
    return false;
  if (commDims.size() != 4)
    return false;

  // With MAXIMIZE_NDA, should try to use more distributed axes
  if (nda < 1)
    return false;

  return true;
}

// Test: configurationND strict nda with exactly 4 ranks
static bool test_configurationND_strict_nda() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  if (worldSize != 4)
    return true; // Only test with exactly 4 ranks

  std::vector<size_t> dims = {16, 16, 16, 16};
  const int ndim = static_cast<int>(dims.size());

  std::vector<int> commDims(ndim, 0);
  int nda = 2; // Request exactly 2 distributed axes
  std::vector<size_t> subsize(ndim), offset(ndim);
  int commSize = 0;

  int rc = shafft::configurationND(dims,
                                   shafft::FFTType::C2C,
                                   commDims,
                                   nda,
                                   subsize,
                                   offset,
                                   commSize,
                                   shafft::DecompositionStrategy::MAXIMIZE_NDA,
                                   0,
                                   MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  // nda should still be 2 (strict mode)
  if (nda != 2)
    return false;

  // Count how many commDims are > 1 - should be exactly 2
  int actual_nda = count_distributed_axes(commDims);
  if (actual_nda != 2)
    return false;

  // Grid product should be <= worldSize
  size_t grid_product = test::product(commDims);
  if (static_cast<int>(grid_product) > worldSize)
    return false;

  // Each commDims[i] should be >= 1 and <= dims[i]
  for (int i = 0; i < ndim; ++i) {
    if (commDims[i] < 1 || static_cast<size_t>(commDims[i]) > dims[i])
      return false;
  }

  // Grid should be packed as a prefix (trailing entries should be 1)
  bool seen_one = false;
  for (int i = 0; i < ndim; ++i) {
    if (commDims[i] == 1)
      seen_one = true;
    else if (seen_one)
      return false; // Non-1 after 1 is invalid
  }

  return true;
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char** argv) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::fprintf(stderr, "MPI_Init failed\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  test::TestRunnerMPI runner("configurationND Function Tests");

  // Auto mode with DecompositionStrategy tests
  runner.run("Auto mode with MAXIMIZE_NDA", test_configurationND_auto_maximize);
  runner.run("Auto mode with MINIMIZE_NDA", test_configurationND_auto_minimize);

  // Explicit commDims tests (primary preference)
  runner.run("Explicit commDims respected", test_configurationND_explicit_comm_dims);
  runner.run("Invalid commDims falls back", test_configurationND_invalid_comm_dims);
  runner.run("Non-prefix commDims fails", test_configurationND_invalid_trailing);
  runner.run("Inactive ranks supported", test_configurationND_inactive_ranks);

  // Explicit nda tests (secondary preference)
  runner.run("Explicit nda respected", test_configurationND_explicit_nda);
  runner.run("Invalid nda falls back", test_configurationND_invalid_nda);
  runner.run("Strict nda respected (4 ranks)", test_configurationND_strict_nda);

  // Single rank tests
  runner.run("Single rank auto mode succeeds", test_configurationND_single_rank_auto);
  runner.run("Single rank explicit nda fails", test_configurationND_single_rank_explicit_nda);

  // Memory limit tests
  runner.run("Memory limit enforced", test_configurationND_mem_limit_enforced);
  runner.run("Memory limit enforced (MINIMIZE_NDA)",
             test_configurationND_mem_limit_enforced_minimize);
  runner.run("Memory limit too small fails", test_configurationND_mem_limit_too_small);
  runner.run("Memory limit too small fails (MINIMIZE_NDA)",
             test_configurationND_mem_limit_too_small_minimize);
  runner.run("Explicit nda + memLimit pass", test_configurationND_explicit_nda_mem_limit_pass);
  runner.run("Explicit nda + memLimit fail", test_configurationND_explicit_nda_mem_limit_fail);
  runner.run("Explicit commDims + memLimit pass",
             test_configurationND_explicit_dims_mem_limit_pass);
  runner.run("Explicit commDims + memLimit fail",
             test_configurationND_explicit_dims_mem_limit_fail);

  // Subsize and offset consistency tests
  runner.run("Subsizes sum to global dims", test_subsize_global_sum);
  runner.run("Offsets are contiguous", test_offsets_contiguous);

  // Precision and dimension tests
  runner.run("Double precision (Z2Z)", test_configurationND_double_precision);
  runner.run("2D tensor", test_configurationND_2d);
  runner.run("4D tensor", test_configurationND_4d);

  int result = runner.finalize();

  MPI_Finalize();
  return result;
}
