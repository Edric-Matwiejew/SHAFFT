/**
 * @file test_initcart.cpp
 * @brief Test initCart with explicit Cartesian process grid
 */
#include "test_utils.hpp"
#include <shafft/shafft.hpp>
#include <mpi.h>
#include <vector>
#include <numeric>

using namespace test;

//------------------------------------------------------------------------------
// Test: Basic initCart with matching grid
//------------------------------------------------------------------------------
static bool test_basic_initcart() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    std::vector<int> dims = {64, 64, 32};
    // Cartesian grid: all ranks on first axis, 1 on others
    // Last axis must be 1 (not distributed)
    std::vector<int> commDims = {world_size, 1, 1};
    
    shafft::Plan plan;
    int rc = plan.initCart(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    
    if (rc != 0) return false;
    if (!plan.isInitialized()) return false;
    
    // Verify we can query the plan
    size_t alloc = plan.allocSize();
    return alloc > 0;
}

//------------------------------------------------------------------------------
// Test: initCart with 2D grid (4 ranks = 2x2x1)
//------------------------------------------------------------------------------
static bool test_2d_grid() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 4) return true;  // Skip unless exactly 4 ranks
    
    std::vector<int> dims = {64, 64, 32};
    std::vector<int> commDims = {2, 2, 1};  // 2x2 grid on first two axes
    
    shafft::Plan plan;
    int rc = plan.initCart(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    
    if (rc != 0) return false;
    
    // Verify layout - first two axes should be distributed
    std::vector<int> subsize(3), offset(3);
    plan.getLayout(subsize, offset, shafft::TensorLayout::INITIAL);
    
    // With 2x2 grid: each rank gets half of dim 0 and half of dim 1
    // But dim 2 should be full
    bool dim2_full = (subsize[2] == dims[2]);
    
    return dim2_full;
}

//------------------------------------------------------------------------------
// Test: initCart result matches equivalent init() result
//------------------------------------------------------------------------------
static bool test_initcart_matches_init() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    std::vector<int> dims = {64, 64, 32};
    
    // Using init with nda=1
    shafft::Plan plan_nda;
    int rc1 = plan_nda.init(1, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc1 != 0) return false;
    
    // Equivalent: commDims = {world_size, 1, 1}
    std::vector<int> commDims = {world_size, 1, 1};
    shafft::Plan plan_cart;
    int rc2 = plan_cart.initCart(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc2 != 0) return false;
    
    // Both should produce same allocSize
    size_t alloc_nda = plan_nda.allocSize();
    size_t alloc_cart = plan_cart.allocSize();
    
    if (alloc_nda != alloc_cart) return false;
    
    // Both should produce same initial layout
    std::vector<int> sub_nda(3), off_nda(3);
    std::vector<int> sub_cart(3), off_cart(3);
    
    plan_nda.getLayout(sub_nda, off_nda, shafft::TensorLayout::INITIAL);
    plan_cart.getLayout(sub_cart, off_cart, shafft::TensorLayout::INITIAL);
    
    return (sub_nda == sub_cart && off_nda == off_cart);
}

//------------------------------------------------------------------------------
// Test: 2D tensor with initCart
//------------------------------------------------------------------------------
static bool test_2d_initcart() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    std::vector<int> dims = {128, 64};
    std::vector<int> commDims = {world_size, 1};
    
    shafft::Plan plan;
    int rc = plan.initCart(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    
    if (rc != 0) return false;
    
    std::vector<int> subsize(2), offset(2);
    plan.getLayout(subsize, offset, shafft::TensorLayout::INITIAL);
    
    // Second axis should be full
    return subsize[1] == dims[1];
}

//------------------------------------------------------------------------------
// Test: initCart with double precision
//------------------------------------------------------------------------------
static bool test_initcart_double() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    std::vector<int> dims = {32, 32, 32};
    std::vector<int> commDims = {world_size, 1, 1};
    
    shafft::Plan plan;
    int rc = plan.initCart(commDims, dims, shafft::FFTType::Z2Z, MPI_COMM_WORLD);
    
    if (rc != 0) return false;
    
    // Double precision should use twice the memory per element
    // but allocSize is in elements, so it should be same
    size_t alloc = plan.allocSize();
    
    return alloc > 0;
}

//------------------------------------------------------------------------------
// Test: Execute works after initCart
//------------------------------------------------------------------------------
static bool test_initcart_execute() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    std::vector<int> dims = {32, 32, 32};
    std::vector<int> commDims = {world_size, 1, 1};
    
    shafft::Plan plan;
    int rc = plan.initCart(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    size_t n = plan.allocSize();
    shafft::complexf *data, *work;
    shafft::allocBuffer(n, &data);
    shafft::allocBuffer(n, &work);
    
    std::vector<shafft::complexf> original(n);
    for (size_t i = 0; i < n; ++i) {
        original[i] = {static_cast<float>(i % 7 + 1), 0.0f};
    }
    shafft::copyToBuffer(data, original.data(), n);
    plan.setBuffers(data, work);
    
    // Forward + backward + normalize
    plan.execute(shafft::FFTDirection::FORWARD);
    plan.execute(shafft::FFTDirection::BACKWARD);
    plan.normalize();
    
    // Get result
    shafft::complexf *result_data, *result_work;
    plan.getBuffers(&result_data, &result_work);
    
    std::vector<shafft::complexf> result(n);
    shafft::copyFromBuffer(result.data(), result_data, n);
    
    // Verify recovery
    bool match = true;
    for (size_t i = 0; i < n; ++i) {
        if (!approx_eq(result[i], original[i], 1e-4f)) {
            match = false;
            break;
        }
    }
    
    shafft::freeBuffer(data);
    shafft::freeBuffer(work);
    
    return match;
}

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    TestRunner runner("initCart Tests");
    
    runner.run("basic_initcart", test_basic_initcart);
    runner.run("2d_grid", test_2d_grid);
    runner.run("initcart_matches_init", test_initcart_matches_init);
    runner.run("2d_initcart", test_2d_initcart);
    runner.run("initcart_double", test_initcart_double);
    runner.run("initcart_execute", test_initcart_execute);
    
    int result = runner.finalize();
    MPI_Finalize();
    return result;
}
