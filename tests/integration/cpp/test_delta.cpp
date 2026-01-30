/**
 * @file test_delta.cpp
 * @brief Test FFT of delta function produces expected constant output
 */
#include "test_utils.hpp"
#include <shafft/shafft.hpp>
#include <mpi.h>
#include <vector>
#include <cmath>

using namespace test;

// Test that FFT of delta at origin gives constant value
static bool test_delta_3d() {
    std::vector<int> dims = {64, 64, 32};
    const int nda = 1;
    
    shafft::Plan plan;
    int rc = plan.init(nda, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    std::vector<int> subsize(3), offset(3);
    plan.getLayout(subsize, offset, shafft::TensorLayout::CURRENT);
    
    size_t local_elems = product(subsize);
    size_t alloc_elems = plan.allocSize();
    if (alloc_elems == 0) return false;
    
    shafft::complexf *data = nullptr, *work = nullptr;
    shafft::allocBuffer(alloc_elems, &data);
    shafft::allocBuffer(alloc_elems, &work);
    
    // Initialize: delta at origin
    std::vector<shafft::complexf> host(alloc_elems, {0.0f, 0.0f});
    
    // Check if origin is on this rank
    bool has_origin = (offset[0] == 0 && offset[1] == 0 && offset[2] == 0);
    if (has_origin) {
        host[0] = {1.0f, 0.0f};
    }
    
    shafft::copyToBuffer(data, host.data(), alloc_elems);
    plan.setBuffers(data, work);
    
    // Forward FFT
    rc = plan.execute(shafft::FFTDirection::FORWARD);
    if (rc != 0) { shafft::freeBuffer(data); shafft::freeBuffer(work); return false; }
    
    // Normalize
    rc = plan.normalize();
    if (rc != 0) { shafft::freeBuffer(data); shafft::freeBuffer(work); return false; }
    
    // Get result
    shafft::complexf *result_data, *result_work;
    plan.getBuffers(&result_data, &result_work);
    
    std::vector<shafft::complexf> result(alloc_elems);
    shafft::copyFromBuffer(result.data(), result_data, alloc_elems);
    
    // Expected: constant value 1/sqrt(N) everywhere
    size_t N = product(dims);
    float expected_val = 1.0f / std::sqrt(static_cast<float>(N));
    shafft::complexf expected = {expected_val, 0.0f};
    
    // Get transformed layout
    std::vector<int> trans_subsize(3), trans_offset(3);
    plan.getLayout(trans_subsize, trans_offset, shafft::TensorLayout::CURRENT);
    size_t trans_local = product(trans_subsize);
    
    bool passed = true;
    for (size_t i = 0; i < trans_local; ++i) {
        if (!approx_eq(result[i], expected, 1e-4f)) {
            passed = false;
            break;
        }
    }
    
    shafft::freeBuffer(data);
    shafft::freeBuffer(work);
    
    return passed;
}

// Test unnormalized FFT of delta gives all ones
static bool test_delta_unnormalized() {
    std::vector<int> dims = {32, 32};
    const int nda = 1;
    
    shafft::Plan plan;
    int rc = plan.init(nda, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    std::vector<int> subsize(2), offset(2);
    plan.getLayout(subsize, offset, shafft::TensorLayout::CURRENT);
    
    size_t alloc_elems = plan.allocSize();
    
    shafft::complexf *data = nullptr, *work = nullptr;
    shafft::allocBuffer(alloc_elems, &data);
    shafft::allocBuffer(alloc_elems, &work);
    
    // Delta at origin
    std::vector<shafft::complexf> host(alloc_elems, {0.0f, 0.0f});
    bool has_origin = (offset[0] == 0 && offset[1] == 0);
    if (has_origin) host[0] = {1.0f, 0.0f};
    
    shafft::copyToBuffer(data, host.data(), alloc_elems);
    plan.setBuffers(data, work);
    
    // Forward FFT without normalize
    rc = plan.execute(shafft::FFTDirection::FORWARD);
    if (rc != 0) { shafft::freeBuffer(data); shafft::freeBuffer(work); return false; }
    
    shafft::complexf *result_data, *result_work;
    plan.getBuffers(&result_data, &result_work);
    
    std::vector<shafft::complexf> result(alloc_elems);
    shafft::copyFromBuffer(result.data(), result_data, alloc_elems);
    
    std::vector<int> trans_subsize(2), trans_offset(2);
    plan.getLayout(trans_subsize, trans_offset, shafft::TensorLayout::CURRENT);
    size_t trans_local = product(trans_subsize);
    
    // Expected: all ones (unnormalized)
    shafft::complexf expected = {1.0f, 0.0f};
    
    bool passed = true;
    for (size_t i = 0; i < trans_local; ++i) {
        if (!approx_eq(result[i], expected, 1e-5f)) {
            passed = false;
            break;
        }
    }
    
    shafft::freeBuffer(data);
    shafft::freeBuffer(work);
    
    return passed;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    TestRunner runner("Delta Function Tests");
    
    runner.run("3D delta -> constant (normalized)", test_delta_3d);
    runner.run("2D delta -> ones (unnormalized)", test_delta_unnormalized);
    
    int result = runner.finalize();
    MPI_Finalize();
    return result;
}
