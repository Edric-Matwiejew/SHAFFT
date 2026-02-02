/**
 * @file test_roundtrip.cpp
 * @brief Test that forward + backward + normalize recovers original data
 */
#include "test_utils.hpp"
#include <shafft/shafft.hpp>
#include <mpi.h>
#include <vector>
#include <cmath>

using namespace test;

// Test round-trip for a given dimension configuration
static bool test_roundtrip_dims(const std::vector<int>& dims, int nda) {
    shafft::Plan plan;
    int rc = plan.init(nda, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    // Get layout
    std::vector<int> subsize(dims.size()), offset(dims.size());
    (void)plan.getLayout(subsize, offset, shafft::TensorLayout::CURRENT);
    
    size_t local_elems = product(subsize);
    size_t alloc_elems = plan.allocSize();
    if (alloc_elems == 0) return false;
    
    // Allocate
    shafft::complexf *data = nullptr, *work = nullptr;
    (void)shafft::allocBuffer(alloc_elems, &data);
    (void)shafft::allocBuffer(alloc_elems, &work);
    
    // Initialize with pattern based on global position
    std::vector<shafft::complexf> original(alloc_elems, {0.0f, 0.0f});
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Simple pattern: delta at origin
    for (size_t i = 0; i < local_elems; ++i) {
        // Check if this is the origin
        bool is_origin = true;
        size_t idx = i;
        for (int d = (int)dims.size() - 1; d >= 0; --d) {
            int local_coord = idx % subsize[d];
            int global_coord = offset[d] + local_coord;
            if (global_coord != 0) is_origin = false;
            idx /= subsize[d];
        }
        original[i] = is_origin ? shafft::complexf{1.0f, 0.0f} 
                                : shafft::complexf{0.0f, 0.0f};
    }
    
    (void)shafft::copyToBuffer(data, original.data(), alloc_elems);
    (void)plan.setBuffers(data, work);
    
    // Forward
    rc = plan.execute(shafft::FFTDirection::FORWARD);
    if (rc != 0) { (void)shafft::freeBuffer(data); (void)shafft::freeBuffer(work); return false; }
    
    // Normalize after forward (symmetric scaling)
    rc = plan.normalize();
    if (rc != 0) { (void)shafft::freeBuffer(data); (void)shafft::freeBuffer(work); return false; }
    
    // Backward
    rc = plan.execute(shafft::FFTDirection::BACKWARD);
    if (rc != 0) { (void)shafft::freeBuffer(data); (void)shafft::freeBuffer(work); return false; }
    
    // Normalize after backward
    rc = plan.normalize();
    if (rc != 0) { (void)shafft::freeBuffer(data); (void)shafft::freeBuffer(work); return false; }
    
    // Get final data
    shafft::complexf *final_data, *final_work;
    (void)plan.getBuffers(&final_data, &final_work);
    
    // Verify layout is back to initial
    std::vector<int> final_subsize(dims.size()), final_offset(dims.size());
    (void)plan.getLayout(final_subsize, final_offset, shafft::TensorLayout::CURRENT);
    
    for (size_t d = 0; d < dims.size(); ++d) {
        if (final_subsize[d] != subsize[d] || final_offset[d] != offset[d]) {
            (void)shafft::freeBuffer(data); (void)shafft::freeBuffer(work);
            return false;
        }
    }
    
    // Copy back and compare
    std::vector<shafft::complexf> result(alloc_elems);
    (void)shafft::copyFromBuffer(result.data(), final_data, alloc_elems);
    
    bool passed = true;
    for (size_t i = 0; i < local_elems; ++i) {
        if (!approx_eq(result[i], original[i], 1e-4f)) {
            passed = false;
            break;
        }
    }
    
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    
    return passed;
}

// Test cases
static bool test_3d_nda1() { return test_roundtrip_dims({64, 64, 32}, 1); }
static bool test_3d_nda2() { return test_roundtrip_dims({32, 32, 32}, 2); }
static bool test_2d_nda1() { return test_roundtrip_dims({128, 64}, 1); }
static bool test_4d_nda1() { return test_roundtrip_dims({16, 16, 16, 8}, 1); }
static bool test_odd_dims() { return test_roundtrip_dims({63, 47, 31}, 1); }

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    TestRunner runner("Round-trip Tests");
    
    runner.run("3D NDA=1 [64x64x32]", test_3d_nda1);
    runner.run("3D NDA=2 [32x32x32]", test_3d_nda2);
    runner.run("2D NDA=1 [128x64]", test_2d_nda1);
    runner.run("4D NDA=1 [16x16x16x8]", test_4d_nda1);
    runner.run("Odd dimensions [63x47x31]", test_odd_dims);
    
    int result = runner.finalize();
    MPI_Finalize();
    return result;
}
