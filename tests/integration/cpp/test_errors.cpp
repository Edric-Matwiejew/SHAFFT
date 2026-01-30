/**
 * @file test_errors.cpp
 * @brief Test that invalid inputs return correct error codes
 */
#include "test_utils.hpp"
#include <shafft/shafft.hpp>
#include <mpi.h>
#include <vector>

using namespace test;

// Helper to convert Status enum to int for comparison
constexpr int S(shafft::Status s) { return static_cast<int>(s); }

// Test that 1D tensor with nda > 0 returns error (only with multi-rank)
// Note: With single rank, nda is forced to 0, so this test only makes sense with 2+ ranks
static bool test_1d_invalid_nda() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // With single rank, the library allows any nda (forces to 0)
    // This test is only meaningful with multiple ranks
    if (world_size == 1) return true;  // Skip for single rank
    
    shafft::Plan plan;
    std::vector<int> dims = {64};  // 1D
    
    // nda = 1 should fail for 1D (can't distribute)
    int rc = plan.init(1, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    
    // Should return SHAFFT_ERR_INVALID_DECOMP
    return rc == S(shafft::Status::SHAFFT_ERR_INVALID_DECOMP);
}

// Test that nda >= ndim returns error (need at least 1 contiguous axis)
static bool test_nda_too_large() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // With single rank, nda is forced to 0
    if (world_size == 1) return true;
    
    shafft::Plan plan;
    std::vector<int> dims = {32, 32};  // 2D
    
    // nda = 2 should fail for 2D (need at least 1 contiguous axis)
    int rc = plan.init(2, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    
    return rc == S(shafft::Status::SHAFFT_ERR_INVALID_DECOMP);
}

// Test that zero dimension returns error
static bool test_zero_dim() {
    shafft::Plan plan;
    std::vector<int> dims = {64, 0, 32};
    
    int rc = plan.init(1, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    
    return rc != S(shafft::Status::SHAFFT_SUCCESS);
}

// Test that negative dimension returns error
static bool test_negative_dim() {
    shafft::Plan plan;
    std::vector<int> dims = {64, -1, 32};
    
    int rc = plan.init(1, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    
    return rc != S(shafft::Status::SHAFFT_SUCCESS);
}

// Test that empty dims returns error
static bool test_empty_dims() {
    shafft::Plan plan;
    std::vector<int> dims = {};
    
    int rc = plan.init(0, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    
    return rc != S(shafft::Status::SHAFFT_SUCCESS);
}

// Test execute without setBuffers returns error
static bool test_execute_no_buffers() {
    shafft::Plan plan;
    std::vector<int> dims = {32, 32};
    
    int rc = plan.init(1, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != S(shafft::Status::SHAFFT_SUCCESS)) return false;
    
    // Don't set buffers, try to execute
    rc = plan.execute(shafft::FFTDirection::FORWARD);
    
    return rc == S(shafft::Status::SHAFFT_ERR_NO_BUFFER);
}

// Test setBuffers with null pointers
static bool test_null_buffers() {
    shafft::Plan plan;
    std::vector<int> dims = {32, 32};
    
    int rc = plan.init(1, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != S(shafft::Status::SHAFFT_SUCCESS)) return false;
    
    shafft::complexf* null_ptr = nullptr;
    rc = plan.setBuffers(null_ptr, null_ptr);
    
    return rc == S(shafft::Status::SHAFFT_ERR_NULLPTR);
}

// Test double release is safe
static bool test_double_release() {
    shafft::Plan plan;
    std::vector<int> dims = {32, 32};
    
    int rc = plan.init(1, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != S(shafft::Status::SHAFFT_SUCCESS)) return false;
    
    plan.release();
    plan.release();  // Should not crash
    
    return true;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    TestRunner runner("Error Handling Tests");
    
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
