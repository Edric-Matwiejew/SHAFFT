/**
 * @file test_configuration_modes.cpp
 * @brief Tests for configurationNDA and configurationCart functions
 * 
 * Tests cover:
 * - configurationNDA: auto mode (nda=0), specific nda, want_max/want_min
 * - configurationCart: auto mode (COMM_DIMS zeros), explicit COMM_DIMS, want_max/want_min
 * - Edge cases: single rank, invalid inputs, inactive ranks
 */

#include "test_utils.hpp"
#include <shafft/shafft.hpp>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstring>

using namespace test;

// Return the number of prime factors (with multiplicity) of n
static int prime_factor_count(int n) {
    int count = 0;
    for (int p = 2; p * p <= n; ++p) {
        while (n % p == 0) {
            ++count;
            n /= p;
        }
    }
    if (n > 1) ++count;
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

// ============================================================================
// configurationNDA tests
// ============================================================================

// Test: configurationNDA with specific nda is strictly respected
// When nda is explicitly specified, the returned nda must match exactly.
// This is consistent with configurationCart's behavior with explicit COMM_DIMS.
static bool test_configurationNDA_specific_nda() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 4) return true;  // Skip if not enough ranks

    std::vector<int> dims = {16, 16, 16, 16};  // 4D tensor
    const int ndim = 4;
    int requested_nda = 2;  // Request exactly 2 distributed axes
    int nda = requested_nda;
    
    std::vector<int> subsize(ndim), offset(ndim), COMM_DIMS(ndim);
    
    int rc = shafft::configurationNDA(dims, nda, subsize, offset, COMM_DIMS,
                                       shafft::FFTType::C2C, 0, MPI_COMM_WORLD);

    // If the world size cannot support two distributed axes (>1 each),
    // configurationNDA should now fail with INVALID_DECOMP.
    if (prime_factor_count(world_size) < requested_nda) {
        return expect_status(rc, shafft::Status::SHAFFT_ERR_INVALID_DECOMP);
    }
    if (rc != 0) return false;
    
    // nda should be exactly what was requested (strict mode)
    if (nda != requested_nda) return false;
    
    // Count how many COMM_DIMS are > 1
    int actual_nda = 0;
    for (int i = 0; i < ndim; ++i) {
        if (COMM_DIMS[i] > 1) actual_nda++;
    }
    
    // The returned nda should match the count of >1 entries
    if (actual_nda != nda) return false;
    
    // Verify COMM_DIMS forms a valid prefix (>1 entries are at the front)
    bool seen_one = false;
    for (int i = 0; i < ndim; ++i) {
        if (COMM_DIMS[i] == 1) seen_one = true;
        else if (seen_one) return false;  // Non-1 after 1 is invalid prefix
    }
    
    // Grid product should equal world_size or be less (inactive ranks)
    int grid_product = test::product(COMM_DIMS);
    if (grid_product > world_size) return false;
    
    return true;
}

// Test: configurationNDA auto mode (nda=0) with want_max (mem_limit >= 0)
static bool test_configurationNDA_auto_want_max() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 2) return true;  // Skip single rank
    
    std::vector<int> dims = {32, 32, 32};
    const int ndim = 3;
    int nda = 0;  // Auto mode
    
    std::vector<int> subsize(ndim), offset(ndim), COMM_DIMS(ndim);
    
    // mem_limit = 0 means want_max (maximize distribution)
    int rc = shafft::configurationNDA(dims, nda, subsize, offset, COMM_DIMS,
                                       shafft::FFTType::C2C, 0, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    // nda should be set to some value >= 1
    if (nda < 1) return false;
    
    // Verify nda matches actual COMM_DIMS distribution count
    int actual_nda = 0;
    for (int i = 0; i < ndim; ++i) {
        if (COMM_DIMS[i] > 1) actual_nda++;
    }
    if (actual_nda != nda) return false;
    
    // Verify grid product is reasonable
    int grid_product = test::product(COMM_DIMS);
    if (grid_product <= 0 || grid_product > world_size) return false;
    
    // Verify subsizes are positive and within bounds
    for (int i = 0; i < ndim; ++i) {
        if (subsize[i] <= 0 || subsize[i] > dims[i]) return false;
    }
    
    return true;
}

// Test: configurationNDA auto mode with want_min (mem_limit < 0)
static bool test_configurationNDA_auto_want_min() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 2) return true;
    
    std::vector<int> dims = {32, 32, 32};
    const int ndim = 3;
    int nda_max = 0;
    int nda_min = 0;
    
    std::vector<int> subsize_max(ndim), offset_max(ndim), COMM_DIMS_max(ndim);
    std::vector<int> subsize_min(ndim), offset_min(ndim), COMM_DIMS_min(ndim);
    
    // First get want_max result
    int rc = shafft::configurationNDA(dims, nda_max, subsize_max, offset_max, COMM_DIMS_max,
                                       shafft::FFTType::C2C, 0, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    // Now get want_min result (mem_limit < 0)
    rc = shafft::configurationNDA(dims, nda_min, subsize_min, offset_min, COMM_DIMS_min,
                                   shafft::FFTType::C2C, -1, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    // want_min should give nda <= want_max's nda
    if (nda_min > nda_max) return false;
    
    // Both should be valid (>= 1)
    if (nda_min < 1 || nda_max < 1) return false;
    
    // Verify both have valid grid products
    int prod_max = test::product(COMM_DIMS_max);
    int prod_min = test::product(COMM_DIMS_min);
    if (prod_max <= 0 || prod_max > world_size) return false;
    if (prod_min <= 0 || prod_min > world_size) return false;
    
    return true;
}

// Test: configurationNDA single rank case
static bool test_configurationNDA_single_rank() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size > 1) return true;  // Only test with single rank
    
    std::vector<int> dims = {64, 64, 32};
    const int ndim = 3;
    int nda = 1;  // Request distribution, but single rank
    
    std::vector<int> subsize(ndim), offset(ndim), COMM_DIMS(ndim);
    
    int rc = shafft::configurationNDA(dims, nda, subsize, offset, COMM_DIMS,
                                       shafft::FFTType::C2C, 0, MPI_COMM_WORLD);
    // With strict explicit mode, single-rank cannot satisfy nda>0; expect INVALID_DECOMP
    return expect_status(rc, shafft::Status::SHAFFT_ERR_INVALID_DECOMP);
}

// Test: configurationNDA invalid nda (>= ndim) should fail
static bool test_configurationNDA_invalid_nda() {
    std::vector<int> dims = {64, 64, 32};
    const int ndim = 3;
    int nda = 3;  // Invalid: nda must be < ndim
    
    std::vector<int> subsize(ndim), offset(ndim), COMM_DIMS(ndim);
    
    int rc = shafft::configurationNDA(dims, nda, subsize, offset, COMM_DIMS,
                                       shafft::FFTType::C2C, 0, MPI_COMM_WORLD);
    
    // Should fail
    return expect_status(rc, shafft::Status::SHAFFT_ERR_INVALID_DECOMP);
}

// ============================================================================
// configurationCart tests
// ============================================================================

// Test: configurationCart with explicit COMM_DIMS
static bool test_configurationCart_explicit_grid() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 4) return true;
    
    std::vector<int> dims = {64, 64, 32};
    const int ndim = 3;
    
    // Request [2, 2, 1] grid (4 ranks total)
    std::vector<int> COMM_DIMS = {2, 2, 1};
    std::vector<int> subsize(ndim), offset(ndim);
    int COMM_SIZE = 0;
    
    int rc = shafft::configurationCart(dims, subsize, offset, COMM_DIMS, COMM_SIZE,
                                        shafft::FFTType::C2C, 0, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    // COMM_SIZE should be 4
    if (COMM_SIZE != 4) return false;
    
    // COMM_DIMS should be preserved
    if (COMM_DIMS[0] != 2 || COMM_DIMS[1] != 2 || COMM_DIMS[2] != 1) return false;
    
    return true;
}

// Test: configurationCart with all-zeros COMM_DIMS (auto mode)
static bool test_configurationCart_auto_zeros() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 2) return true;
    
    std::vector<int> dims = {64, 64, 32};
    const int ndim = 3;
    
    // All zeros -> auto compute
    std::vector<int> COMM_DIMS = {0, 0, 0};
    std::vector<int> subsize(ndim), offset(ndim);
    int COMM_SIZE = 0;
    
    int rc = shafft::configurationCart(dims, subsize, offset, COMM_DIMS, COMM_SIZE,
                                        shafft::FFTType::C2C, 0, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    // COMM_DIMS should be filled in with valid values
    int prod = test::product(COMM_DIMS);
    if (prod <= 0 || prod > world_size) return false;
    if (COMM_SIZE != prod) return false;
    
    // Each COMM_DIMS[i] should be >= 1 and <= dims[i]
    for (int i = 0; i < ndim; ++i) {
        if (COMM_DIMS[i] < 1 || COMM_DIMS[i] > dims[i]) return false;
    }
    
    // Subsizes should be positive and within bounds
    for (int i = 0; i < ndim; ++i) {
        if (subsize[i] <= 0 || subsize[i] > dims[i]) return false;
    }
    
    // Grid should be a valid prefix (no gaps)
    bool seen_one = false;
    for (int i = 0; i < ndim; ++i) {
        if (COMM_DIMS[i] == 1) seen_one = true;
        else if (seen_one) return false;
    }
    
    return true;
}

// Test: configurationCart auto mode with want_max vs want_min
static bool test_configurationCart_auto_modes() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 4) return true;
    
    std::vector<int> dims = {32, 32, 32, 32};  // 4D
    const int ndim = 4;
    
    // want_max (mem_limit = 0)
    std::vector<int> COMM_DIMS_max = {0, 0, 0, 0};
    std::vector<int> subsize_max(ndim), offset_max(ndim);
    int COMM_SIZE_max = 0;
    
    int rc = shafft::configurationCart(dims, subsize_max, offset_max, COMM_DIMS_max, COMM_SIZE_max,
                                        shafft::FFTType::C2C, 0, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    // Count distributed axes for max
    int nda_max = 0;
    for (int i = 0; i < ndim; ++i) {
        if (COMM_DIMS_max[i] > 1) nda_max++;
    }
    
    // want_min (mem_limit < 0)
    std::vector<int> COMM_DIMS_min = {0, 0, 0, 0};
    std::vector<int> subsize_min(ndim), offset_min(ndim);
    int COMM_SIZE_min = 0;
    
    rc = shafft::configurationCart(dims, subsize_min, offset_min, COMM_DIMS_min, COMM_SIZE_min,
                                    shafft::FFTType::C2C, -1, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    // Count distributed axes for min
    int nda_min = 0;
    for (int i = 0; i < ndim; ++i) {
        if (COMM_DIMS_min[i] > 1) nda_min++;
    }
    
    // want_min should have nda <= want_max's nda
    if (nda_min > nda_max) return false;
    
    // Both should have valid nda (>= 1 for multi-rank)
    if (nda_min < 1 || nda_max < 1) return false;
    
    // Verify COMM_SIZE matches grid product
    if (COMM_SIZE_max != test::product(COMM_DIMS_max)) return false;
    if (COMM_SIZE_min != test::product(COMM_DIMS_min)) return false;
    
    // Verify both have reasonable COMM_SIZE
    if (COMM_SIZE_max <= 0 || COMM_SIZE_max > world_size) return false;
    if (COMM_SIZE_min <= 0 || COMM_SIZE_min > world_size) return false;
    
    return true;
}

// Test: configurationCart with invalid COMM_DIMS (product > world_size)
static bool test_configurationCart_invalid_grid() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    std::vector<int> dims = {64, 64, 32};
    const int ndim = 3;
    
    // Request grid that uses more ranks than available
    std::vector<int> COMM_DIMS = {world_size + 1, 1, 1};
    std::vector<int> subsize(ndim), offset(ndim);
    int COMM_SIZE = 0;
    
    int rc = shafft::configurationCart(dims, subsize, offset, COMM_DIMS, COMM_SIZE,
                                        shafft::FFTType::C2C, 0, MPI_COMM_WORLD);
    
    // Should fail
    return expect_status(rc, shafft::Status::SHAFFT_ERR_INVALID_DECOMP);
}

// Test: configurationCart with COMM_DIMS that uses fewer ranks (inactive ranks)
static bool test_configurationCart_inactive_ranks() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 5) return true;  // Need more ranks than we'll use
    
    std::vector<int> dims = {64, 64, 32};
    const int ndim = 3;
    
    // Request [2, 2, 1] grid (4 ranks) with 5+ available
    std::vector<int> COMM_DIMS = {2, 2, 1};
    std::vector<int> subsize(ndim), offset(ndim);
    int COMM_SIZE = 0;
    
    int rc = shafft::configurationCart(dims, subsize, offset, COMM_DIMS, COMM_SIZE,
                                        shafft::FFTType::C2C, 0, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    // COMM_SIZE should be 4 (not world_size)
    if (COMM_SIZE != 4) return false;
    
    return true;
}

// Test: configurationCart single rank with all ones
static bool test_configurationCart_single_rank() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size > 1) return true;  // Only test with single rank
    
    std::vector<int> dims = {64, 64, 32};
    const int ndim = 3;
    
    // Single rank: [1, 1, 1]
    std::vector<int> COMM_DIMS = {1, 1, 1};
    std::vector<int> subsize(ndim), offset(ndim);
    int COMM_SIZE = 0;
    
    int rc = shafft::configurationCart(dims, subsize, offset, COMM_DIMS, COMM_SIZE,
                                        shafft::FFTType::C2C, 0, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    // Full tensor
    for (int i = 0; i < ndim; ++i) {
        if (subsize[i] != dims[i]) return false;
        if (offset[i] != 0) return false;
    }
    if (COMM_SIZE != 1) return false;
    
    return true;
}

// Test: configurationCart grid with trailing >1 (invalid)
static bool test_configurationCart_invalid_trailing() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 4) return true;
    
    std::vector<int> dims = {64, 64, 32};
    const int ndim = 3;
    
    // Invalid: last axis must be 1 for valid slab decomposition
    std::vector<int> COMM_DIMS = {1, 2, 2};  // Invalid pattern
    std::vector<int> subsize(ndim), offset(ndim);
    int COMM_SIZE = 0;
    
    int rc = shafft::configurationCart(dims, subsize, offset, COMM_DIMS, COMM_SIZE,
                                        shafft::FFTType::C2C, 0, MPI_COMM_WORLD);
    
    // Should fail (non-contiguous distributed axes)
    return expect_status(rc, shafft::Status::SHAFFT_ERR_INVALID_DECOMP);
}

// Test: Verify subsizes sum correctly across ranks
static bool test_subsize_global_sum() {
    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (world_size < 2) return true;
    
    std::vector<int> dims = {100, 50, 25};  // Non-power-of-2
    const int ndim = 3;
    int nda = 1;
    
    std::vector<int> subsize(ndim), offset(ndim), COMM_DIMS(ndim);
    
    int rc = shafft::configurationNDA(dims, nda, subsize, offset, COMM_DIMS,
                                       shafft::FFTType::C2C, 0, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    // For each distributed axis, subsizes should sum to global dim
    for (int axis = 0; axis < nda; ++axis) {
        if (COMM_DIMS[axis] > 1) {
            int sum = global_sum(subsize[axis]);
            if (sum != dims[axis]) return false;
        }
    }
    
    return true;
}

// Test: NDA and Cart produce consistent results
static bool test_nda_cart_consistency() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // 2D tensor for simpler comparison
    std::vector<int> dims = {64, 64};
    
    // NDA configuration (explicit nda = 1)
    int nda = 1;
    std::vector<int> nda_subsize(2), nda_offset(2), nda_comm_dims(2);
    int rc1 = shafft::configurationNDA(dims, nda, nda_subsize, nda_offset, nda_comm_dims,
                                       shafft::FFTType::C2C, 0, MPI_COMM_WORLD);
    if (world_size == 1) {
        return expect_status(rc1, shafft::Status::SHAFFT_ERR_INVALID_DECOMP);
    }
    if (rc1 != 0) return false;
    
    // Cart configuration using the grid chosen by configurationNDA
    std::vector<int> cart_subsize(2), cart_offset(2);
    std::vector<int> cart_comm_dims = nda_comm_dims;
    int comm_size = 0;
    int rc2 = shafft::configurationCart(dims, cart_subsize, cart_offset, cart_comm_dims,
                                        comm_size, shafft::FFTType::C2C, 0, MPI_COMM_WORLD);
    if (rc2 != 0) return false;
    
    // Local layouts should match
    return (nda_subsize == cart_subsize && nda_offset == cart_offset);
}

// Test: Offsets form contiguous blocks across active ranks
static bool test_offsets_contiguous() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    std::vector<int> dims = {100, 50};
    int nda = 1;
    std::vector<int> subsize(2), offset(2), COMM_DIMS(2);
    
    int rc = shafft::configurationNDA(dims, nda, subsize, offset, COMM_DIMS,
                                       shafft::FFTType::C2C, 0, MPI_COMM_WORLD);
    if (world_size == 1) {
        return expect_status(rc, shafft::Status::SHAFFT_ERR_INVALID_DECOMP);
    }
    if (rc != 0) return false;
    
    // Gather offsets/sizes for axis 0 from all ranks
    std::vector<int> all_offsets(world_size), all_sizes(world_size);
    MPI_Allgather(&offset[0], 1, MPI_INT, all_offsets.data(), 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&subsize[0], 1, MPI_INT, all_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    // Only first cart_size ranks are active
    int cart_size = COMM_DIMS[0];
    if (cart_size < 1) return false;
    
    for (int i = 0; i < cart_size - 1; ++i) {
        if (all_offsets[i] + all_sizes[i] != all_offsets[i + 1]) {
            return false;
        }
    }
    
    // Last active rank should end exactly at global dimension
    if (all_offsets[cart_size - 1] + all_sizes[cart_size - 1] != dims[0]) {
        return false;
    }
    
    return true;
}

// Test: configurationNDA with explicit nda=2 produces exactly 2 distributed axes
// With strict mode, the grid must have exactly nda axes with >1 entries
static bool test_configurationNDA_strict_nda() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 4) return true;  // Only test with exactly 4 ranks
    
    std::vector<int> dims = {16, 16, 16, 16};
    const int ndim = 4;
    int nda = 2;  // Request exactly 2 distributed axes
    
    std::vector<int> subsize(ndim), offset(ndim), COMM_DIMS(ndim);
    
    int rc = shafft::configurationNDA(dims, nda, subsize, offset, COMM_DIMS,
                                       shafft::FFTType::C2C, 0, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    // nda should still be 2 (strict mode)
    if (nda != 2) return false;
    
    // Count how many COMM_DIMS are > 1 - should be exactly 2
    int actual_nda = 0;
    for (int i = 0; i < ndim; ++i) {
        if (COMM_DIMS[i] > 1) actual_nda++;
    }
    if (actual_nda != 2) return false;
    
    // Grid product should be <= world_size
    int grid_product = test::product(COMM_DIMS);
    if (grid_product > world_size) return false;
    
    // Each COMM_DIMS[i] should be >= 1 and <= dims[i]
    for (int i = 0; i < ndim; ++i) {
        if (COMM_DIMS[i] < 1 || COMM_DIMS[i] > dims[i]) return false;
    }
    
    // Grid should be packed as a prefix (trailing entries should be 1)
    bool seen_one = false;
    for (int i = 0; i < ndim; ++i) {
        if (COMM_DIMS[i] == 1) seen_one = true;
        else if (seen_one) return false;  // Non-1 after 1 is invalid
    }
    
    return true;
}

// ============================================================================
// Memory limit tests
// ============================================================================

// Test: configurationNDA with mem_limit enforces size constraint
static bool test_configurationNDA_mem_limit_enforced() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 2) return true;  // Need multiple ranks
    
    // 64x64x64 C2C tensor = 64^3 * 8 bytes = 2MB per buffer
    // Per rank footprints:
    //   no distribution: 2MB per buffer -> 4MB for data+work (exceeds limit)
    //   2-way slab on axis 0: 1MB per buffer -> 2MB for data+work (fits)
    std::vector<int> dims = {64, 64, 64};
    const int ndim = 3;
    int nda = 0;  // Auto mode
    
    std::vector<int> subsize(ndim), offset(ndim), COMM_DIMS(ndim);
    
    // Set a memory limit that requires distribution
    // after splitting across two ranks.
    size_t mem_limit = 2 * 1024 * 1024;  // 2MB
    
    int rc = shafft::configurationNDA(dims, nda, subsize, offset, COMM_DIMS,
                                       shafft::FFTType::C2C, mem_limit, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    // Verify local size is within limit
    size_t local_size = 1;
    for (int i = 0; i < ndim; ++i) {
        local_size *= subsize[i];
    }
    local_size *= 8;  // C2C = 2 * sizeof(float) = 8 bytes
    // Account for data + work buffers
    local_size *= 2;
    
    if (local_size > mem_limit) return false;
    
    return true;
}

// Test: configurationNDA fails when mem_limit is too small
static bool test_configurationNDA_mem_limit_too_small() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 2) return true;  // Need multiple ranks for meaningful test
    
    // 64x64x64 C2C tensor with very small mem_limit
    // Even with maximum distribution, minimum local size is bounded
    std::vector<int> dims = {64, 64, 64};
    const int ndim = 3;
    int nda = 0;  // Auto mode
    
    std::vector<int> subsize(ndim), offset(ndim), COMM_DIMS(ndim);
    
    // Set impossibly small memory limit (1 byte)
    size_t mem_limit = 1;
    
    int rc = shafft::configurationNDA(dims, nda, subsize, offset, COMM_DIMS,
                                       shafft::FFTType::C2C, mem_limit, MPI_COMM_WORLD);
    
    // Should fail - no valid decomposition fits in 1 byte
    return expect_status(rc, shafft::Status::SHAFFT_ERR_INVALID_DECOMP);
}

// Test: configurationCart with mem_limit enforces size constraint
static bool test_configurationCart_mem_limit_enforced() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 2) return true;
    
    std::vector<int> dims = {64, 64, 64};
    const int ndim = 3;
    
    // Auto mode for Cart
    std::vector<int> COMM_DIMS = {0, 0, 0};
    std::vector<int> subsize(ndim), offset(ndim);
    int COMM_SIZE = 0;
    
    // Memory limit that requires distribution
    size_t mem_limit = 2 * 1024 * 1024;  // 2MB
    
    int rc = shafft::configurationCart(dims, subsize, offset, COMM_DIMS, COMM_SIZE,
                                        shafft::FFTType::C2C, mem_limit, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    // Verify local size is within limit
    size_t local_size = 1;
    for (int i = 0; i < ndim; ++i) {
        local_size *= subsize[i];
    }
    local_size *= 8;  // C2C = 8 bytes
    local_size *= 2;  // data + work
    
    if (local_size > mem_limit) return false;
    
    return true;
}

// Test: configurationCart fails when mem_limit is too small for explicit grid
static bool test_configurationCart_mem_limit_explicit_fail() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 2) return true;
    
    std::vector<int> dims = {64, 64, 64};
    const int ndim = 3;
    
    // Explicit grid with minimal distribution
    std::vector<int> COMM_DIMS = {2, 1, 1};
    std::vector<int> subsize(ndim), offset(ndim);
    int COMM_SIZE = 0;
    
    // Very small memory limit - should fail with this explicit grid
    size_t mem_limit = 1;
    
    int rc = shafft::configurationCart(dims, subsize, offset, COMM_DIMS, COMM_SIZE,
                                        shafft::FFTType::C2C, mem_limit, MPI_COMM_WORLD);
    
    // Should fail - explicit grid doesn't meet memory constraint
    return expect_status(rc, shafft::Status::SHAFFT_ERR_INVALID_DECOMP);
}

// ============================================================================
// Additional coverage for all mode combinations
// ============================================================================

// Test: configurationNDA explicit nda with want_min (mem_limit < 0)
static bool test_configurationNDA_explicit_want_min() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 2) return true;
    
    std::vector<int> dims = {32, 32, 32};
    const int ndim = 3;
    int nda = 1;  // Explicit: request exactly 1 distributed axis
    
    std::vector<int> subsize(ndim), offset(ndim), COMM_DIMS(ndim);
    
    // want_min mode (mem_limit < 0), but explicit nda should still be respected
    int rc = shafft::configurationNDA(dims, nda, subsize, offset, COMM_DIMS,
                                       shafft::FFTType::C2C, -1, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    // nda should remain 1 (explicit mode is strict)
    if (nda != 1) return false;
    
    return true;
}

// Test: configurationNDA explicit nda with mem_limit enforced (pass case)
static bool test_configurationNDA_explicit_mem_limit_pass() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 2) return true;
    
    std::vector<int> dims = {32, 32, 32};
    const int ndim = 3;
    int nda = 1;  // Explicit nda
    
    std::vector<int> subsize(ndim), offset(ndim), COMM_DIMS(ndim);
    
    // Large enough mem_limit that nda=1 should work
    // 32*32*32 / 2 = 16384 elements * 8 bytes = 131072 bytes per rank
    size_t mem_limit = 256 * 1024;  // 256KB - should be enough
    
    int rc = shafft::configurationNDA(dims, nda, subsize, offset, COMM_DIMS,
                                       shafft::FFTType::C2C, mem_limit, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    // Verify local size is within limit
    size_t local_size = 1;
    for (int i = 0; i < ndim; ++i) local_size *= subsize[i];
    local_size *= 8;
    local_size *= 2;  // data + work buffers
    
    if (local_size > mem_limit) return false;
    
    return true;
}

// Test: configurationNDA explicit nda with mem_limit enforced (fail case)
static bool test_configurationNDA_explicit_mem_limit_fail() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 2) return true;
    
    std::vector<int> dims = {64, 64, 64};
    const int ndim = 3;
    int nda = 1;  // Explicit nda=1 gives less distribution
    
    std::vector<int> subsize(ndim), offset(ndim), COMM_DIMS(ndim);
    
    // Very small mem_limit - nda=1 won't be enough distribution
    size_t mem_limit = 1;
    
    int rc = shafft::configurationNDA(dims, nda, subsize, offset, COMM_DIMS,
                                       shafft::FFTType::C2C, mem_limit, MPI_COMM_WORLD);
    
    // Should fail - explicit nda=1 doesn't meet memory constraint
    return expect_status(rc, shafft::Status::SHAFFT_ERR_INVALID_DECOMP);
}

// Test: configurationCart explicit grid with want_min (mem_limit < 0)
static bool test_configurationCart_explicit_want_min() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 2) return true;
    
    std::vector<int> dims = {32, 32, 32};
    const int ndim = 3;
    
    std::vector<int> COMM_DIMS = {2, 1, 1};  // Explicit grid
    std::vector<int> subsize(ndim), offset(ndim);
    int COMM_SIZE = 0;
    
    // want_min mode, but explicit grid should be respected exactly
    int rc = shafft::configurationCart(dims, subsize, offset, COMM_DIMS, COMM_SIZE,
                                        shafft::FFTType::C2C, -1, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    // Grid should be unchanged
    if (COMM_DIMS[0] != 2 || COMM_DIMS[1] != 1 || COMM_DIMS[2] != 1) return false;
    
    return true;
}

// Test: configurationCart explicit grid with mem_limit enforced (pass case)
static bool test_configurationCart_explicit_mem_limit_pass() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 2) return true;
    
    std::vector<int> dims = {32, 32, 32};
    const int ndim = 3;
    
    std::vector<int> COMM_DIMS = {2, 1, 1};
    std::vector<int> subsize(ndim), offset(ndim);
    int COMM_SIZE = 0;
    
    // Large mem_limit that should pass
    size_t mem_limit = 256 * 1024;
    
    int rc = shafft::configurationCart(dims, subsize, offset, COMM_DIMS, COMM_SIZE,
                                        shafft::FFTType::C2C, mem_limit, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    // Verify local size is within limit
    size_t local_size = 1;
    for (int i = 0; i < ndim; ++i) local_size *= subsize[i];
    local_size *= 8;
    local_size *= 2;  // data + work
    
    if (local_size > mem_limit) return false;
    
    return true;
}

// Test: configurationCart auto mode with mem_limit too small
static bool test_configurationCart_mem_limit_too_small() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 2) return true;
    
    std::vector<int> dims = {64, 64, 64};
    const int ndim = 3;
    
    std::vector<int> COMM_DIMS = {0, 0, 0};  // Auto mode
    std::vector<int> subsize(ndim), offset(ndim);
    int COMM_SIZE = 0;
    
    // Impossibly small mem_limit
    size_t mem_limit = 1;
    
    int rc = shafft::configurationCart(dims, subsize, offset, COMM_DIMS, COMM_SIZE,
                                        shafft::FFTType::C2C, mem_limit, MPI_COMM_WORLD);
    
    // Should fail - no grid can satisfy 1 byte limit
    return expect_status(rc, shafft::Status::SHAFFT_ERR_INVALID_DECOMP);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    TestRunner runner("Configuration Function Tests");
    
    // configurationNDA tests
    runner.run("configurationNDA: specific nda succeeds", test_configurationNDA_specific_nda);
    runner.run("configurationNDA: auto mode want_max", test_configurationNDA_auto_want_max);
    runner.run("configurationNDA: auto mode want_min", test_configurationNDA_auto_want_min);
    runner.run("configurationNDA: single rank", test_configurationNDA_single_rank);
    runner.run("configurationNDA: invalid nda fails", test_configurationNDA_invalid_nda);
    runner.run("configurationNDA: strict nda respected", test_configurationNDA_strict_nda);
    
    // configurationCart tests
    runner.run("configurationCart: explicit grid", test_configurationCart_explicit_grid);
    runner.run("configurationCart: auto (all zeros)", test_configurationCart_auto_zeros);
    runner.run("configurationCart: auto want_max vs want_min", test_configurationCart_auto_modes);
    runner.run("configurationCart: invalid grid fails", test_configurationCart_invalid_grid);
    runner.run("configurationCart: inactive ranks", test_configurationCart_inactive_ranks);
    runner.run("configurationCart: single rank", test_configurationCart_single_rank);
    runner.run("configurationCart: invalid trailing fails", test_configurationCart_invalid_trailing);
    
    // Common behavior tests
    runner.run("Subsizes sum to global dims", test_subsize_global_sum);
    runner.run("NDA and Cart consistency", test_nda_cart_consistency);
    runner.run("Offsets are contiguous", test_offsets_contiguous);
    
    // Memory limit tests (auto mode)
    runner.run("configurationNDA: auto mem_limit enforced", test_configurationNDA_mem_limit_enforced);
    runner.run("configurationNDA: auto mem_limit too small fails", test_configurationNDA_mem_limit_too_small);
    runner.run("configurationCart: auto mem_limit enforced", test_configurationCart_mem_limit_enforced);
    runner.run("configurationCart: auto mem_limit too small fails", test_configurationCart_mem_limit_too_small);
    
    // Explicit mode + mem_limit combinations
    runner.run("configurationNDA: explicit + want_min", test_configurationNDA_explicit_want_min);
    runner.run("configurationNDA: explicit + mem_limit pass", test_configurationNDA_explicit_mem_limit_pass);
    runner.run("configurationNDA: explicit + mem_limit fail", test_configurationNDA_explicit_mem_limit_fail);
    runner.run("configurationCart: explicit + want_min", test_configurationCart_explicit_want_min);
    runner.run("configurationCart: explicit + mem_limit pass", test_configurationCart_explicit_mem_limit_pass);
    runner.run("configurationCart: explicit + mem_limit fail", test_configurationCart_mem_limit_explicit_fail);
    
    int result = runner.finalize();
    
    MPI_Finalize();
    return result;
}
