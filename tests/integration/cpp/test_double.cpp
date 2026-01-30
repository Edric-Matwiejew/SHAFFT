/**
 * @file test_double.cpp
 * @brief Test double precision (Z2Z) transforms
 */
#include "test_utils.hpp"
#include <shafft/shafft.hpp>
#include <mpi.h>
#include <vector>
#include <cmath>

using namespace test;

// Test round-trip for double precision
static bool test_z2z_roundtrip() {
    std::vector<int> dims = {64, 64, 32};
    const int nda = 1;
    
    shafft::Plan plan;
    int rc = plan.init(nda, dims, shafft::FFTType::Z2Z, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    std::vector<int> subsize(3), offset(3);
    plan.getLayout(subsize, offset, shafft::TensorLayout::CURRENT);
    
    size_t local_elems = product(subsize);
    size_t alloc_elems = plan.allocSize();
    if (alloc_elems == 0) return false;
    
    shafft::complexd *data = nullptr, *work = nullptr;
    shafft::allocBuffer(alloc_elems, &data);
    shafft::allocBuffer(alloc_elems, &work);
    
    // Initialize: delta at origin
    std::vector<shafft::complexd> original(alloc_elems, {0.0, 0.0});
    
    for (size_t i = 0; i < local_elems; ++i) {
        bool is_origin = true;
        size_t idx = i;
        for (int d = 2; d >= 0; --d) {
            int local_coord = idx % subsize[d];
            int global_coord = offset[d] + local_coord;
            if (global_coord != 0) is_origin = false;
            idx /= subsize[d];
        }
        original[i] = is_origin ? shafft::complexd{1.0, 0.0} 
                                : shafft::complexd{0.0, 0.0};
    }
    
    shafft::copyToBuffer(data, original.data(), alloc_elems);
    plan.setBuffers(data, work);
    
    // Forward + normalize + backward + normalize
    rc = plan.execute(shafft::FFTDirection::FORWARD);
    if (rc != 0) { shafft::freeBuffer(data); shafft::freeBuffer(work); return false; }
    
    rc = plan.normalize();
    if (rc != 0) { shafft::freeBuffer(data); shafft::freeBuffer(work); return false; }
    
    rc = plan.execute(shafft::FFTDirection::BACKWARD);
    if (rc != 0) { shafft::freeBuffer(data); shafft::freeBuffer(work); return false; }
    
    rc = plan.normalize();
    if (rc != 0) { shafft::freeBuffer(data); shafft::freeBuffer(work); return false; }
    
    shafft::complexd *final_data, *final_work;
    plan.getBuffers(&final_data, &final_work);
    
    std::vector<shafft::complexd> result(alloc_elems);
    shafft::copyFromBuffer(result.data(), final_data, alloc_elems);
    
    // Compare with tighter tolerance for double
    bool passed = true;
    for (size_t i = 0; i < local_elems; ++i) {
        if (!approx_eq(result[i], original[i], 1e-10)) {
            passed = false;
            break;
        }
    }
    
    shafft::freeBuffer(data);
    shafft::freeBuffer(work);
    
    return passed;
}

// Test double precision delta -> constant
static bool test_z2z_delta() {
    std::vector<int> dims = {32, 32, 32};
    const int nda = 1;
    
    shafft::Plan plan;
    int rc = plan.init(nda, dims, shafft::FFTType::Z2Z, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    std::vector<int> subsize(3), offset(3);
    plan.getLayout(subsize, offset, shafft::TensorLayout::CURRENT);
    
    size_t alloc_elems = plan.allocSize();
    
    shafft::complexd *data = nullptr, *work = nullptr;
    shafft::allocBuffer(alloc_elems, &data);
    shafft::allocBuffer(alloc_elems, &work);
    
    std::vector<shafft::complexd> host(alloc_elems, {0.0, 0.0});
    bool has_origin = (offset[0] == 0 && offset[1] == 0 && offset[2] == 0);
    if (has_origin) host[0] = {1.0, 0.0};
    
    shafft::copyToBuffer(data, host.data(), alloc_elems);
    plan.setBuffers(data, work);
    
    rc = plan.execute(shafft::FFTDirection::FORWARD);
    if (rc != 0) { shafft::freeBuffer(data); shafft::freeBuffer(work); return false; }
    
    rc = plan.normalize();
    if (rc != 0) { shafft::freeBuffer(data); shafft::freeBuffer(work); return false; }
    
    shafft::complexd *result_data, *result_work;
    plan.getBuffers(&result_data, &result_work);
    
    std::vector<shafft::complexd> result(alloc_elems);
    shafft::copyFromBuffer(result.data(), result_data, alloc_elems);
    
    size_t N = product(dims);
    double expected_val = 1.0 / std::sqrt(static_cast<double>(N));
    shafft::complexd expected = {expected_val, 0.0};
    
    std::vector<int> trans_subsize(3), trans_offset(3);
    plan.getLayout(trans_subsize, trans_offset, shafft::TensorLayout::CURRENT);
    size_t trans_local = product(trans_subsize);
    
    bool passed = true;
    for (size_t i = 0; i < trans_local; ++i) {
        if (!approx_eq(result[i], expected, 1e-10)) {
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
    
    TestRunner runner("Double Precision (Z2Z) Tests");
    
    runner.run("Z2Z round-trip", test_z2z_roundtrip);
    runner.run("Z2Z delta -> constant", test_z2z_delta);
    
    int result = runner.finalize();
    MPI_Finalize();
    return result;
}
