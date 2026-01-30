/**
 * @file test_layout.cpp
 * @brief Test that getLayout returns correct subsize/offset for various configurations
 */
#include "test_utils.hpp"
#include <shafft/shafft.hpp>
#include <mpi.h>
#include <vector>
#include <numeric>

using namespace test;

// Verify that all ranks' subsizes sum to global dims
static bool test_subsize_sum() {
    std::vector<int> dims = {64, 64, 32};
    const int nda = 1;
    
    shafft::Plan plan;
    int rc = plan.init(nda, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    std::vector<int> subsize(3), offset(3);
    plan.getLayout(subsize, offset, shafft::TensorLayout::INITIAL);
    
    // Gather all subsizes for axis 0 (distributed axis)
    int local_size0 = subsize[0];
    int total_size0 = 0;
    MPI_Allreduce(&local_size0, &total_size0, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    // Should equal global dimension
    if (total_size0 != dims[0]) return false;
    
    // Non-distributed axes should equal global dims
    if (subsize[1] != dims[1]) return false;
    if (subsize[2] != dims[2]) return false;
    
    return true;
}

// Verify offsets are contiguous (no gaps, no overlaps)
static bool test_offsets_contiguous() {
    std::vector<int> dims = {100, 50};  // Non-power-of-2 to test balanced decomposition
    const int nda = 1;
    
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    shafft::Plan plan;
    int rc = plan.init(nda, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    std::vector<int> subsize(2), offset(2);
    plan.getLayout(subsize, offset, shafft::TensorLayout::INITIAL);
    
    // Gather all (offset, subsize) pairs
    std::vector<int> all_offsets(world_size), all_sizes(world_size);
    MPI_Allgather(&offset[0], 1, MPI_INT, all_offsets.data(), 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&subsize[0], 1, MPI_INT, all_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    // Verify contiguity: offset[i] + size[i] == offset[i+1]
    for (int i = 0; i < world_size - 1; ++i) {
        if (all_offsets[i] + all_sizes[i] != all_offsets[i + 1]) {
            return false;
        }
    }
    
    // Last rank should reach the end
    if (all_offsets[world_size - 1] + all_sizes[world_size - 1] != dims[0]) {
        return false;
    }
    
    return true;
}

// Test TRANSFORMED layout is different from INITIAL for NDA >= 1
static bool test_transformed_layout() {
    std::vector<int> dims = {64, 64, 32};
    const int nda = 1;
    
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 2) return true;  // Skip with single rank
    
    shafft::Plan plan;
    int rc = plan.init(nda, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    std::vector<int> init_subsize(3), init_offset(3);
    std::vector<int> trans_subsize(3), trans_offset(3);
    
    plan.getLayout(init_subsize, init_offset, shafft::TensorLayout::INITIAL);
    plan.getLayout(trans_subsize, trans_offset, shafft::TensorLayout::TRANSFORMED);
    
    // For nda=1, distribution should shift from axis 0 to last axis
    // Initial: axis 0 is distributed (subsize[0] < dims[0])
    // Transformed: last axis is distributed (subsize[2] < dims[2])
    
    bool init_dist_0 = (init_subsize[0] < dims[0]);
    bool trans_dist_2 = (trans_subsize[2] < dims[2]);
    
    return init_dist_0 && trans_dist_2;
}

// Test CURRENT tracks execute() calls
static bool test_current_tracks_state() {
    std::vector<int> dims = {32, 32, 32};
    const int nda = 1;
    
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 2) return true;
    
    shafft::Plan plan;
    int rc = plan.init(nda, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    // Before execute: CURRENT == INITIAL
    std::vector<int> init_sub(3), init_off(3);
    std::vector<int> curr_sub(3), curr_off(3);
    
    plan.getLayout(init_sub, init_off, shafft::TensorLayout::INITIAL);
    plan.getLayout(curr_sub, curr_off, shafft::TensorLayout::CURRENT);
    
    for (int i = 0; i < 3; ++i) {
        if (init_sub[i] != curr_sub[i] || init_off[i] != curr_off[i]) {
            return false;
        }
    }
    
    // Allocate and execute forward
    size_t n = plan.allocSize();
    shafft::complexf *data, *work;
    shafft::allocBuffer(n, &data);
    shafft::allocBuffer(n, &work);
    
    std::vector<shafft::complexf> host(n, {0.0f, 0.0f});
    shafft::copyToBuffer(data, host.data(), n);
    plan.setBuffers(data, work);
    
    plan.execute(shafft::FFTDirection::FORWARD);
    
    // After forward: CURRENT == TRANSFORMED
    std::vector<int> trans_sub(3), trans_off(3);
    plan.getLayout(trans_sub, trans_off, shafft::TensorLayout::TRANSFORMED);
    plan.getLayout(curr_sub, curr_off, shafft::TensorLayout::CURRENT);
    
    bool match_trans = true;
    for (int i = 0; i < 3; ++i) {
        if (trans_sub[i] != curr_sub[i] || trans_off[i] != curr_off[i]) {
            match_trans = false;
        }
    }
    
    shafft::freeBuffer(data);
    shafft::freeBuffer(work);
    
    return match_trans;
}

//------------------------------------------------------------------------------
// INITIAL Layout Tests
//------------------------------------------------------------------------------

// Test: INITIAL layout distributes first nda axes
static bool test_initial_distributed_axes() {
    std::vector<int> dims = {64, 64, 32};
    const int ndim = static_cast<int>(dims.size());
    const int nda = 1;
    
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    shafft::Plan plan;
    int rc = plan.init(nda, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    std::vector<int> subsize(ndim), offset(ndim);
    plan.getLayout(subsize, offset, shafft::TensorLayout::INITIAL);
    
    if (world_size > 1) {
        // First nda axes should be distributed (subsize < global)
        for (int i = 0; i < nda; ++i) {
            if (subsize[i] >= dims[i]) return false;  // Should be less than global
        }
        // Remaining axes should be full (subsize == global)
        for (int i = nda; i < ndim; ++i) {
            if (subsize[i] != dims[i]) return false;
        }
    } else {
        // Single rank: all axes full
        for (int i = 0; i < ndim; ++i) {
            if (subsize[i] != dims[i]) return false;
        }
    }
    
    return true;
}

// Test: INITIAL layout with 2 distributed axes (nda=2)
static bool test_initial_nda2() {
    std::vector<int> dims = {16, 16, 16, 16};  // 4D tensor
    const int ndim = static_cast<int>(dims.size());
    const int nda = 2;
    
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 4) return true;  // Need at least 4 ranks for 2D decomposition
    
    shafft::Plan plan;
    int rc = plan.init(nda, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    std::vector<int> subsize(ndim), offset(ndim);
    plan.getLayout(subsize, offset, shafft::TensorLayout::INITIAL);
    
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
    std::vector<int> dims = {64, 64, 32};
    const int ndim = static_cast<int>(dims.size());
    const int nda = 1;
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    shafft::Plan plan;
    int rc = plan.init(nda, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    std::vector<int> subsize(ndim), offset(ndim);
    plan.getLayout(subsize, offset, shafft::TensorLayout::INITIAL);
    
    if (rank == 0) {
        // Rank 0 should have offset 0 for all axes
        for (int i = 0; i < ndim; ++i) {
            if (offset[i] != 0) return false;
        }
    }
    
    // All ranks: non-distributed axes should have offset 0
    for (int i = nda; i < ndim; ++i) {
        if (offset[i] != 0) return false;
    }
    
    return true;
}

// Test: INITIAL layout with single rank (no distribution)
static bool test_initial_single_rank() {
    std::vector<int> dims = {64, 64, 32};
    const int ndim = static_cast<int>(dims.size());
    const int nda = 1;  // Request distribution, but single rank means none
    
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size > 1) return true;  // Only test with single rank
    
    shafft::Plan plan;
    int rc = plan.init(nda, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    std::vector<int> subsize(ndim), offset(ndim);
    plan.getLayout(subsize, offset, shafft::TensorLayout::INITIAL);
    
    // Single rank: subsize == dims, offset == 0
    for (int i = 0; i < ndim; ++i) {
        if (subsize[i] != dims[i]) return false;
        if (offset[i] != 0) return false;
    }
    
    return true;
}

// Test: INITIAL layout is consistent across ranks (global view matches)
static bool test_initial_global_consistency() {
    std::vector<int> dims = {100, 50, 25};  // Non-power-of-2
    const int ndim = static_cast<int>(dims.size());
    const int nda = 1;
    
    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    shafft::Plan plan;
    int rc = plan.init(nda, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    std::vector<int> subsize(ndim), offset(ndim);
    plan.getLayout(subsize, offset, shafft::TensorLayout::INITIAL);
    
    // Gather all offsets and subsizes for distributed axis
    std::vector<int> all_offsets(world_size), all_sizes(world_size);
    MPI_Allgather(&offset[0], 1, MPI_INT, all_offsets.data(), 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&subsize[0], 1, MPI_INT, all_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    // Verify: offset[i] == sum of sizes before rank i
    int expected_offset = 0;
    for (int r = 0; r < world_size; ++r) {
        if (all_offsets[r] != expected_offset) return false;
        expected_offset += all_sizes[r];
    }
    
    // Total should equal global size
    if (expected_offset != dims[0]) return false;
    
    return true;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    TestRunner runner("Layout Tests");
    
    runner.run("Subsizes sum to global dims", test_subsize_sum);
    runner.run("Offsets are contiguous", test_offsets_contiguous);
    runner.run("TRANSFORMED differs from INITIAL", test_transformed_layout);
    runner.run("CURRENT tracks execute() state", test_current_tracks_state);
    
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
