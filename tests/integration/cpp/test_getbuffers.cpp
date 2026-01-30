/**
 * @file test_getbuffers.cpp
 * @brief Test that getBuffers tracks buffer swaps after execute
 */
#include "test_utils.hpp"
#include <shafft/shafft.hpp>
#include <mpi.h>
#include <vector>

using namespace test;

//------------------------------------------------------------------------------
// Test: getBuffers returns what was set via setBuffers
//------------------------------------------------------------------------------
static bool test_get_matches_set() {
    std::vector<int> dims = {32, 32};
    const int nda = 1;
    
    shafft::Plan plan;
    int rc = plan.init(nda, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    size_t n = plan.allocSize();
    shafft::complexf *data, *work;
    shafft::allocBuffer(n, &data);
    shafft::allocBuffer(n, &work);
    
    rc = plan.setBuffers(data, work);
    if (rc != 0) {
        shafft::freeBuffer(data);
        shafft::freeBuffer(work);
        return false;
    }
    
    // Get buffers back
    shafft::complexf *got_data, *got_work;
    rc = plan.getBuffers(&got_data, &got_work);
    if (rc != 0) {
        shafft::freeBuffer(data);
        shafft::freeBuffer(work);
        return false;
    }
    
    // Before any execute, should match what was set
    bool match = (got_data == data && got_work == work);
    
    shafft::freeBuffer(data);
    shafft::freeBuffer(work);
    
    return match;
}

//------------------------------------------------------------------------------
// Test: Buffers may swap after forward execute
//------------------------------------------------------------------------------
static bool test_buffers_swap_after_forward() {
    std::vector<int> dims = {32, 32, 32};
    const int nda = 1;
    
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    shafft::Plan plan;
    int rc = plan.init(nda, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    size_t n = plan.allocSize();
    shafft::complexf *data, *work;
    shafft::allocBuffer(n, &data);
    shafft::allocBuffer(n, &work);
    
    // Initialize data
    std::vector<shafft::complexf> host(n, {1.0f, 0.0f});
    shafft::copyToBuffer(data, host.data(), n);
    plan.setBuffers(data, work);
    
    shafft::complexf *before_data, *before_work;
    plan.getBuffers(&before_data, &before_work);
    
    // Execute forward
    plan.execute(shafft::FFTDirection::FORWARD);
    
    shafft::complexf *after_data, *after_work;
    plan.getBuffers(&after_data, &after_work);
    
    // After execute, data pointer should be one of the original buffers
    // (either unchanged or swapped with work)
    bool valid_data = (after_data == data || after_data == work);
    bool valid_work = (after_work == data || after_work == work);
    
    // Data and work should be different
    bool different = (after_data != after_work);
    
    shafft::freeBuffer(data);
    shafft::freeBuffer(work);
    
    return valid_data && valid_work && different;
}

//------------------------------------------------------------------------------
// Test: Data buffer contains result after forward
//------------------------------------------------------------------------------
static bool test_data_buffer_has_result() {
    std::vector<int> dims = {16, 16};
    const int nda = 1;
    
    shafft::Plan plan;
    int rc = plan.init(nda, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    size_t n = plan.allocSize();
    shafft::complexf *data, *work;
    shafft::allocBuffer(n, &data);
    shafft::allocBuffer(n, &work);
    
    // Initialize with non-zero data
    std::vector<shafft::complexf> host(n);
    for (size_t i = 0; i < n; ++i) {
        host[i] = {static_cast<float>(i % 10), 0.0f};
    }
    shafft::copyToBuffer(data, host.data(), n);
    
    // Clear work buffer
    std::vector<shafft::complexf> zeros(n, {0.0f, 0.0f});
    shafft::copyToBuffer(work, zeros.data(), n);
    
    plan.setBuffers(data, work);
    
    // Execute forward
    plan.execute(shafft::FFTDirection::FORWARD);
    
    // Get current data buffer
    shafft::complexf *result_data, *result_work;
    plan.getBuffers(&result_data, &result_work);
    
    // Read result back
    std::vector<shafft::complexf> result(n);
    shafft::copyFromBuffer(result.data(), result_data, n);
    
    // Result should be non-trivial (not all zeros)
    bool has_nonzero = false;
    for (size_t i = 0; i < n; ++i) {
        if (std::fabs(result[i].real()) > 1e-10f || 
            std::fabs(result[i].imag()) > 1e-10f) {
            has_nonzero = true;
            break;
        }
    }
    
    shafft::freeBuffer(data);
    shafft::freeBuffer(work);
    
    return has_nonzero;
}

//------------------------------------------------------------------------------
// Test: Round-trip uses correct buffers
//------------------------------------------------------------------------------
static bool test_roundtrip_buffer_tracking() {
    std::vector<int> dims = {32, 32};
    const int nda = 1;
    
    shafft::Plan plan;
    int rc = plan.init(nda, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    size_t n = plan.allocSize();
    shafft::complexf *data, *work;
    shafft::allocBuffer(n, &data);
    shafft::allocBuffer(n, &work);
    
    // Initialize with known data
    std::vector<shafft::complexf> original(n);
    for (size_t i = 0; i < n; ++i) {
        original[i] = {static_cast<float>(i % 17 + 1), 
                       static_cast<float>((i * 3) % 13)};
    }
    shafft::copyToBuffer(data, original.data(), n);
    plan.setBuffers(data, work);
    
    // Forward
    plan.execute(shafft::FFTDirection::FORWARD);
    
    // Backward
    plan.execute(shafft::FFTDirection::BACKWARD);
    
    // Normalize
    plan.normalize();
    
    // Get final data buffer
    shafft::complexf *final_data, *final_work;
    plan.getBuffers(&final_data, &final_work);
    
    // Read result
    std::vector<shafft::complexf> result(n);
    shafft::copyFromBuffer(result.data(), final_data, n);
    
    // Verify recovery (with tolerance)
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
// Test: Double precision buffer tracking
//------------------------------------------------------------------------------
static bool test_double_buffer_tracking() {
    std::vector<int> dims = {32, 32};
    const int nda = 1;
    
    shafft::Plan plan;
    int rc = plan.init(nda, dims, shafft::FFTType::Z2Z, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    size_t n = plan.allocSize();
    shafft::complexd *data, *work;
    shafft::allocBuffer(n, &data);
    shafft::allocBuffer(n, &work);
    
    std::vector<shafft::complexd> host(n, {1.0, 2.0});
    shafft::copyToBuffer(data, host.data(), n);
    plan.setBuffers(data, work);
    
    shafft::complexd *got_data, *got_work;
    rc = plan.getBuffers(&got_data, &got_work);
    
    bool match = (got_data == data && got_work == work);
    
    shafft::freeBuffer(data);
    shafft::freeBuffer(work);
    
    return rc == 0 && match;
}

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    TestRunner runner("getBuffers Tests");
    
    runner.run("get_matches_set", test_get_matches_set);
    runner.run("buffers_swap_after_forward", test_buffers_swap_after_forward);
    runner.run("data_buffer_has_result", test_data_buffer_has_result);
    runner.run("roundtrip_buffer_tracking", test_roundtrip_buffer_tracking);
    runner.run("double_buffer_tracking", test_double_buffer_tracking);
    
    int result = runner.finalize();
    MPI_Finalize();
    return result;
}
