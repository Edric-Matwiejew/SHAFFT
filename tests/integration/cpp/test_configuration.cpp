/**
 * @file test_configuration.cpp
 * @brief Test configurationNDA and configurationCart helper functions
 */
#include "test_utils.hpp"
#include <shafft/shafft.hpp>
#include <mpi.h>
#include <vector>
#include <numeric>

using namespace test;

// Helper: compare return code to expected SHAFFT status
static bool expect_status(int rc, shafft::Status expected) {
    return rc == static_cast<int>(expected);
}

//------------------------------------------------------------------------------
// Test: configurationNDA returns valid subsize/offset
//------------------------------------------------------------------------------
static bool test_configuration_nda_basic() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::vector<int> dims = {64, 64, 32};
    int nda = 1;
    std::vector<int> subsize(3), offset(3), comm_dims(3);
    size_t mem_limit = 0;  // No limit
    
    int rc = shafft::configurationNDA(dims, nda, subsize, offset, comm_dims,
                                      shafft::FFTType::C2C, mem_limit, MPI_COMM_WORLD);
    if (world_size == 1) {
        // Explicit nda cannot be satisfied on a single rank
        return expect_status(rc, shafft::Status::SHAFFT_ERR_INVALID_DECOMP);
    }
    if (rc != 0) return false;
    
    // Subsize should be positive for all axes
    for (int i = 0; i < 3; ++i) {
        if (subsize[i] <= 0) return false;
    }
    
    // Offset should be non-negative
    for (int i = 0; i < 3; ++i) {
        if (offset[i] < 0) return false;
    }
    
    return true;
}

//------------------------------------------------------------------------------
// Test: Sum of subsizes equals global dims
//------------------------------------------------------------------------------
static bool test_configuration_nda_sum() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::vector<int> dims = {100, 50, 30};  // Non-power-of-2
    int nda = 1;
    std::vector<int> subsize(3), offset(3), comm_dims(3);
    
    int rc = shafft::configurationNDA(dims, nda, subsize, offset, comm_dims,
                                      shafft::FFTType::C2C, 0, MPI_COMM_WORLD);
    if (world_size == 1) {
        return expect_status(rc, shafft::Status::SHAFFT_ERR_INVALID_DECOMP);
    }
    if (rc != 0) return false;
    
    // Gather subsizes from all ranks
    // For distributed axis 0
    int local_size0 = subsize[0];
    int total_size0 = 0;
    MPI_Allreduce(&local_size0, &total_size0, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    if (total_size0 != dims[0]) return false;
    
    // Non-distributed axes should equal global
    if (subsize[1] != dims[1]) return false;
    if (subsize[2] != dims[2]) return false;
    
    return true;
}

//------------------------------------------------------------------------------
// Test: COMM_DIMS product equals comm size
//------------------------------------------------------------------------------
static bool test_configuration_nda_comm_dims() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::vector<int> dims = {64, 64, 32};
    int nda = 1;
    std::vector<int> subsize(3), offset(3), comm_dims(3);
    
    int rc = shafft::configurationNDA(dims, nda, subsize, offset, comm_dims,
                                      shafft::FFTType::C2C, 0, MPI_COMM_WORLD);
    if (world_size == 1) {
        return expect_status(rc, shafft::Status::SHAFFT_ERR_INVALID_DECOMP);
    }
    if (rc != 0) return false;
    
    int product = 1;
    for (int d : comm_dims) product *= d;
    
    return product == world_size;
}

//------------------------------------------------------------------------------
// Test: configurationCart validates grid
//------------------------------------------------------------------------------
static bool test_configuration_cart_basic() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // configurationCart may have issues with single rank in some configs
    // Use 2D for simplicity
    std::vector<int> dims = {64, 64};
    std::vector<int> subsize(2), offset(2);
    std::vector<int> comm_dims = {world_size, 1};
    int comm_size = 0;
    
    int rc = shafft::configurationCart(dims, subsize, offset, comm_dims,
                                        comm_size, shafft::FFTType::C2C, 0,
                                        MPI_COMM_WORLD);
    
    if (rc != 0) return false;
    
    // comm_size should match world_size
    if (comm_size != world_size) return false;
    
    // Subsize should be valid
    for (int i = 0; i < 2; ++i) {
        if (subsize[i] <= 0) return false;
    }
    
    return true;
}

//------------------------------------------------------------------------------
// Test: NDA and Cart produce consistent results
//------------------------------------------------------------------------------
static bool test_nda_cart_consistency() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // Use 2D for simpler testing
    std::vector<int> dims = {64, 64};
    
    // Get NDA configuration
    int nda = 1;
    std::vector<int> nda_subsize(2), nda_offset(2), nda_comm_dims(2);
    int rc1 = shafft::configurationNDA(dims, nda, nda_subsize, nda_offset, 
                                        nda_comm_dims, shafft::FFTType::C2C, 
                                        0, MPI_COMM_WORLD);
    if (world_size == 1) {
        return expect_status(rc1, shafft::Status::SHAFFT_ERR_INVALID_DECOMP);
    }
    if (rc1 != 0) return false;
    
    // Use the COMM_DIMS from NDA for Cart
    std::vector<int> cart_subsize(2), cart_offset(2);
    std::vector<int> cart_comm_dims = nda_comm_dims;
    int comm_size = 0;
    int rc2 = shafft::configurationCart(dims, cart_subsize, cart_offset,
                                         cart_comm_dims, comm_size,
                                         shafft::FFTType::C2C, 0, MPI_COMM_WORLD);
    if (rc2 != 0) return false;
    
    // Results should match
    return (nda_subsize == cart_subsize && nda_offset == cart_offset);
}

//------------------------------------------------------------------------------
// Test: configurationNDA with double precision
//------------------------------------------------------------------------------
static bool test_configuration_double() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::vector<int> dims = {32, 32, 32};
    int nda = 1;
    std::vector<int> subsize(3), offset(3), comm_dims(3);
    
    int rc = shafft::configurationNDA(dims, nda, subsize, offset, comm_dims,
                                      shafft::FFTType::Z2Z, 0, MPI_COMM_WORLD);
    if (world_size == 1) {
        return expect_status(rc, shafft::Status::SHAFFT_ERR_INVALID_DECOMP);
    }
    if (rc != 0) return false;
    
    // Double should still work
    for (int i = 0; i < 3; ++i) {
        if (subsize[i] <= 0) return false;
    }
    
    return true;
}

//------------------------------------------------------------------------------
// Test: 2D configuration
//------------------------------------------------------------------------------
static bool test_configuration_2d() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::vector<int> dims = {128, 64};
    int nda = 1;
    std::vector<int> subsize(2), offset(2), comm_dims(2);
    
    int rc = shafft::configurationNDA(dims, nda, subsize, offset, comm_dims,
                                      shafft::FFTType::C2C, 0, MPI_COMM_WORLD);
    if (world_size == 1) {
        return expect_status(rc, shafft::Status::SHAFFT_ERR_INVALID_DECOMP);
    }
    if (rc != 0) return false;
    
    // Second axis should be full
    if (subsize[1] != dims[1]) return false;
    
    return true;
}

//------------------------------------------------------------------------------
// Test: Offsets are contiguous
//------------------------------------------------------------------------------
static bool test_configuration_offsets_contiguous() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::vector<int> dims = {100, 50};
    int nda = 1;
    std::vector<int> subsize(2), offset(2), comm_dims(2);
    
    int rc = shafft::configurationNDA(dims, nda, subsize, offset, comm_dims,
                                      shafft::FFTType::C2C, 0, MPI_COMM_WORLD);
    if (world_size == 1) {
        return expect_status(rc, shafft::Status::SHAFFT_ERR_INVALID_DECOMP);
    }
    if (rc != 0) return false;
    
    // Gather all (offset, subsize) pairs for axis 0
    std::vector<int> all_offsets(world_size), all_sizes(world_size);
    MPI_Allgather(&offset[0], 1, MPI_INT, all_offsets.data(), 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&subsize[0], 1, MPI_INT, all_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    // Verify contiguity
    for (int i = 0; i < world_size - 1; ++i) {
        if (all_offsets[i] + all_sizes[i] != all_offsets[i + 1]) {
            return false;
        }
    }
    
    // Last rank should reach end
    if (all_offsets[world_size - 1] + all_sizes[world_size - 1] != dims[0]) {
        return false;
    }
    
    return true;
}

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    TestRunner runner("Configuration Tests");
    
    runner.run("configuration_nda_basic", test_configuration_nda_basic);
    runner.run("configuration_nda_sum", test_configuration_nda_sum);
    runner.run("configuration_nda_comm_dims", test_configuration_nda_comm_dims);
    runner.run("configuration_cart_basic", test_configuration_cart_basic);
    runner.run("nda_cart_consistency", test_nda_cart_consistency);
    runner.run("configuration_double", test_configuration_double);
    runner.run("configuration_2d", test_configuration_2d);
    runner.run("configuration_offsets_contiguous", test_configuration_offsets_contiguous);
    
    int result = runner.finalize();
    MPI_Finalize();
    return result;
}
