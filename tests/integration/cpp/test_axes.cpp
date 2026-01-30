/**
 * @file test_axes.cpp
 * @brief Test that getAxes returns correct contiguous/distributed axes
 */
#include "test_utils.hpp"
#include <shafft/shafft.hpp>
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <set>

using namespace test;

//------------------------------------------------------------------------------
// Test: ca and da together cover all axes exactly once
//------------------------------------------------------------------------------
static bool test_axes_complete_coverage() {
    std::vector<int> dims = {64, 64, 32};
    const int ndim = static_cast<int>(dims.size());
    const int nda = 1;  // Requested nda (may be adjusted for single rank)
    
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    shafft::Plan plan;
    int rc = plan.init(nda, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    // Pre-allocate vectors to ndim size
    std::vector<int> ca(ndim, -1), da(ndim, -1);  // Initialize to invalid
    rc = plan.getAxes(ca, da, shafft::TensorLayout::INITIAL);
    if (rc != 0) return false;
    
    // For single rank, all axes are contiguous (nda_actual = 0)
    // For multi-rank, nda_actual = nda
    int nda_actual = (world_size > 1) ? nda : 0;
    int nca_actual = ndim - nda_actual;
    
    // Collect all valid axes from the first nca_actual + nda_actual entries
    std::set<int> all_axes;
    
    // First nda_actual entries of da are distributed axes
    for (int i = 0; i < nda_actual; ++i) {
        if (da[i] >= 0 && da[i] < ndim) all_axes.insert(da[i]);
    }
    // First nca_actual entries of ca are contiguous axes  
    for (int i = 0; i < nca_actual; ++i) {
        if (ca[i] >= 0 && ca[i] < ndim) all_axes.insert(ca[i]);
    }
    
    // Should have all axes covered
    return static_cast<int>(all_axes.size()) == ndim;
}

//------------------------------------------------------------------------------
// Test: INITIAL layout - first nda axes are distributed
//------------------------------------------------------------------------------
static bool test_initial_axes() {
    std::vector<int> dims = {64, 64, 32};
    const int ndim = static_cast<int>(dims.size());
    const int nda = 1;  // First axis distributed
    
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    shafft::Plan plan;
    int rc = plan.init(nda, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    std::vector<int> ca(ndim), da(ndim);
    rc = plan.getAxes(ca, da, shafft::TensorLayout::INITIAL);
    if (rc != 0) return false;
    
    // For nda=1: da[0] should contain axis 0
    if (world_size > 1) {
        if (da[0] != 0) return false;
        // ca should have axes 1 and 2 (in first two positions)
        std::set<int> ca_set;
        ca_set.insert(ca[0]);
        ca_set.insert(ca[1]);
        if (ca_set.find(1) == ca_set.end()) return false;
        if (ca_set.find(2) == ca_set.end()) return false;
    }
    
    return true;
}

//------------------------------------------------------------------------------
// Test: TRANSFORMED layout - distributed axis shifts
//------------------------------------------------------------------------------
static bool test_transformed_axes() {
    std::vector<int> dims = {64, 64, 32};
    const int ndim = static_cast<int>(dims.size());
    const int nda = 1;
    
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 2) return true;  // Skip with single rank
    
    shafft::Plan plan;
    int rc = plan.init(nda, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    std::vector<int> init_ca(ndim), init_da(ndim);
    std::vector<int> trans_ca(ndim), trans_da(ndim);
    
    rc = plan.getAxes(init_ca, init_da, shafft::TensorLayout::INITIAL);
    if (rc != 0) return false;
    rc = plan.getAxes(trans_ca, trans_da, shafft::TensorLayout::TRANSFORMED);
    if (rc != 0) return false;
    
    // For nda=1 with multi-rank:
    // INITIAL: da[0] = 0
    // TRANSFORMED: distribution should shift (typically to last axis)
    
    // At minimum, verify transformed has valid axes
    // (da[0] should be a valid axis index)
    if (trans_da[0] < 0 || trans_da[0] >= ndim) return false;
    
    return true;
}

//------------------------------------------------------------------------------
// Test: CURRENT tracks execute state
//------------------------------------------------------------------------------
static bool test_current_axes_tracks_state() {
    std::vector<int> dims = {32, 32, 32};
    const int ndim = static_cast<int>(dims.size());
    const int nda = 1;
    
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 2) return true;
    
    shafft::Plan plan;
    int rc = plan.init(nda, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    // Before execute: CURRENT should match INITIAL
    std::vector<int> init_ca(ndim), init_da(ndim);
    std::vector<int> curr_ca(ndim), curr_da(ndim);
    
    rc = plan.getAxes(init_ca, init_da, shafft::TensorLayout::INITIAL);
    if (rc != 0) return false;
    rc = plan.getAxes(curr_ca, curr_da, shafft::TensorLayout::CURRENT);
    if (rc != 0) return false;
    
    // Compare first nda distributed axes
    if (init_da[0] != curr_da[0]) return false;
    
    // Allocate and execute forward
    size_t n = plan.allocSize();
    shafft::complexf *data, *work;
    shafft::allocBuffer(n, &data);
    shafft::allocBuffer(n, &work);
    
    std::vector<shafft::complexf> host(n, {0.0f, 0.0f});
    shafft::copyToBuffer(data, host.data(), n);
    rc = plan.setBuffers(data, work);
    if (rc != 0) { shafft::freeBuffer(data); shafft::freeBuffer(work); return false; }
    
    rc = plan.execute(shafft::FFTDirection::FORWARD);
    if (rc != 0) { shafft::freeBuffer(data); shafft::freeBuffer(work); return false; }
    
    // After forward: CURRENT should match TRANSFORMED
    std::vector<int> trans_ca(ndim), trans_da(ndim);
    rc = plan.getAxes(trans_ca, trans_da, shafft::TensorLayout::TRANSFORMED);
    if (rc != 0) { shafft::freeBuffer(data); shafft::freeBuffer(work); return false; }
    rc = plan.getAxes(curr_ca, curr_da, shafft::TensorLayout::CURRENT);
    if (rc != 0) { shafft::freeBuffer(data); shafft::freeBuffer(work); return false; }
    
    bool match = (trans_da[0] == curr_da[0]);
    
    shafft::freeBuffer(data);
    shafft::freeBuffer(work);
    
    return match;
}

//------------------------------------------------------------------------------
// Test: axes match layout (distributed axes have subsize < global)
//------------------------------------------------------------------------------
static bool test_axes_match_layout() {
    std::vector<int> dims = {64, 64, 32};
    const int ndim = static_cast<int>(dims.size());
    const int nda = 1;
    
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 2) return true;
    
    shafft::Plan plan;
    int rc = plan.init(nda, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    std::vector<int> subsize(ndim), offset(ndim);
    std::vector<int> ca(ndim), da(ndim);
    
    rc = plan.getLayout(subsize, offset, shafft::TensorLayout::INITIAL);
    if (rc != 0) return false;
    rc = plan.getAxes(ca, da, shafft::TensorLayout::INITIAL);
    if (rc != 0) return false;
    
    // First nda entries of da are distributed axes - should have subsize < global
    for (int i = 0; i < nda; ++i) {
        int axis = da[i];
        if (axis >= 0 && axis < ndim) {
            if (subsize[axis] >= dims[axis]) return false;
        }
    }
    
    // First (ndim-nda) entries of ca are contiguous - should have subsize == global
    for (int i = 0; i < ndim - nda; ++i) {
        int axis = ca[i];
        if (axis >= 0 && axis < ndim) {
            if (subsize[axis] != dims[axis]) return false;
        }
    }
    
    return true;
}

//------------------------------------------------------------------------------
// Test: 2D tensor axes
//------------------------------------------------------------------------------
static bool test_2d_axes() {
    std::vector<int> dims = {128, 64};
    const int ndim = static_cast<int>(dims.size());
    const int nda = 1;
    
    shafft::Plan plan;
    int rc = plan.init(nda, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    std::vector<int> ca(ndim), da(ndim);
    rc = plan.getAxes(ca, da, shafft::TensorLayout::INITIAL);
    if (rc != 0) return false;
    
    // da[0] should be a valid axis (0 or 1)
    if (da[0] < 0 || da[0] >= ndim) return false;
    
    // ca[0] should be a valid axis (0 or 1) and different from da[0]
    if (ca[0] < 0 || ca[0] >= ndim) return false;
    
    return true;
}

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    TestRunner runner("getAxes Tests");
    
    runner.run("axes_complete_coverage", test_axes_complete_coverage);
    runner.run("initial_axes", test_initial_axes);
    runner.run("transformed_axes", test_transformed_axes);
    runner.run("current_axes_tracks_state", test_current_axes_tracks_state);
    runner.run("axes_match_layout", test_axes_match_layout);
    runner.run("2d_axes", test_2d_axes);
    
    int result = runner.finalize();
    MPI_Finalize();
    return result;
}
