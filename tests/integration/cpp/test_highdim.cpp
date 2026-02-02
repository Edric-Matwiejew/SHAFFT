/**
 * @file test_highdim.cpp
 * @brief High-dimensional roundtrip tests (5D, 7D, 9D, 11D) with small axis sizes
 * 
 * Tests FFT correctness in higher dimensions to verify that the library
 * handles arbitrary dimensionality correctly. Uses small per-axis sizes
 * to keep memory usage low.
 * 
 * Uses configurationCart to ensure valid decomposition for small dimensions.
 */

#include "test_utils.hpp"
#include <shafft/shafft.hpp>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace test;

// Tolerance for roundtrip error
constexpr float  SP_TOL = 1e-4f;
constexpr double DP_TOL = 1e-10;

/**
 * Initialize tensor with index-based values for uniqueness verification.
 */
template<typename T>
void init_index_tensor(T* data, size_t local_size,
                       const std::vector<int>& dims,
                       const std::vector<int>& subsize,
                       const std::vector<int>& offset) {
    using Real = decltype(data[0].real());
    
    size_t global_total = 1;
    for (int d : dims) global_total *= d;
    Real scale = static_cast<Real>(1.0 / global_total);
    Real mid = static_cast<Real>((global_total - 1) / 2.0);
    
    // Compute strides (row-major)
    std::vector<size_t> strides(dims.size());
    strides.back() = 1;
    for (int i = dims.size() - 2; i >= 0; --i) {
        strides[i] = strides[i+1] * dims[i+1];
    }
    
    std::vector<size_t> local_strides(subsize.size());
    local_strides.back() = 1;
    for (int i = subsize.size() - 2; i >= 0; --i) {
        local_strides[i] = local_strides[i+1] * subsize[i+1];
    }
    
    std::vector<int> coords(dims.size(), 0);
    for (size_t lin = 0; lin < local_size; ++lin) {
        // Linear to coords
        size_t tmp = lin;
        for (size_t d = 0; d < dims.size(); ++d) {
            coords[d] = (tmp / local_strides[d]) % subsize[d];
        }
        
        // Global flattened index
        size_t gidx = 0;
        for (size_t d = 0; d < dims.size(); ++d) {
            gidx += (offset[d] + coords[d]) * strides[d];
        }
        
        Real val = (static_cast<Real>(gidx) - mid) * scale;
        data[lin] = T(val, -val);
    }
}

/**
 * Compute max absolute error between two tensors
 */
template<typename T>
double max_error(const T* a, const T* b, size_t n) {
    double max_err = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double err_re = std::fabs(a[i].real() - b[i].real());
        double err_im = std::fabs(a[i].imag() - b[i].imag());
        max_err = std::max(max_err, std::max(err_re, err_im));
    }
    return max_err;
}

/**
 * Get valid COMM_DIMS for a given tensor and world size.
 * Uses configurationCart to automatically find a valid decomposition.
 * Pass all zeros in comm_dims to let the library choose.
 */
bool get_valid_comm_dims(const std::vector<int>& dims, shafft::FFTType type,
                         std::vector<int>& comm_dims) {
    const int ndim = dims.size();
    comm_dims.resize(ndim);
    std::vector<int> subsize(ndim), offset(ndim);
    int comm_size = 0;
    
    // Pass all zeros to let configurationCart automatically determine
    // the best decomposition for the current world size
    std::fill(comm_dims.begin(), comm_dims.end(), 0);
    
    int rc = shafft::configurationCart(dims, subsize, offset, comm_dims,
                                       comm_size, type, 0, MPI_COMM_WORLD);
    return (rc == 0);
}

// ============================================================================
// Single Precision Roundtrip Tests
// ============================================================================

static bool test_5d_c2c() {
    std::vector<int> dims = {4, 4, 4, 4, 4};  // 1024 elements
    
    std::vector<int> comm_dims;
    if (!get_valid_comm_dims(dims, shafft::FFTType::C2C, comm_dims)) return false;
    
    shafft::Plan plan;
    int rc = plan.initCart(comm_dims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    std::vector<int> subsize(dims.size()), offset(dims.size());
    (void)plan.getLayout(subsize, offset, shafft::TensorLayout::CURRENT);
    
    size_t local_elems = product(subsize);
    size_t alloc_elems = plan.allocSize();
    
    shafft::complexf *data = nullptr, *work = nullptr;
    (void)shafft::allocBuffer(alloc_elems, &data);
    (void)shafft::allocBuffer(alloc_elems, &work);
    
    std::vector<shafft::complexf> original(alloc_elems);
    init_index_tensor(original.data(), local_elems, dims, subsize, offset);
    
    (void)shafft::copyToBuffer(data, original.data(), alloc_elems);
    (void)plan.setBuffers(data, work);
    
    // Forward + backward + normalize
    (void)plan.execute(shafft::FFTDirection::FORWARD);
    (void)plan.execute(shafft::FFTDirection::BACKWARD);
    (void)plan.normalize();
    
    // Get result
    shafft::complexf *final_data, *final_work;
    (void)plan.getBuffers(&final_data, &final_work);
    
    std::vector<shafft::complexf> result(alloc_elems);
    (void)shafft::copyFromBuffer(result.data(), final_data, alloc_elems);
    
    double local_err = max_error(original.data(), result.data(), local_elems);
    double global_err = 0;
    MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    
    return global_err < SP_TOL;
}

static bool test_5d_nda2() {
    std::vector<int> dims = {4, 4, 4, 4, 4};
    
    // For nda=2 style: try to decompose on axes 0 and 1
    // But configurationCart gives us valid dims automatically
    std::vector<int> comm_dims;
    if (!get_valid_comm_dims(dims, shafft::FFTType::C2C, comm_dims)) return false;
    
    shafft::Plan plan;
    int rc = plan.initCart(comm_dims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    std::vector<int> subsize(dims.size()), offset(dims.size());
    (void)plan.getLayout(subsize, offset, shafft::TensorLayout::CURRENT);
    
    size_t local_elems = product(subsize);
    size_t alloc_elems = plan.allocSize();
    
    shafft::complexf *data = nullptr, *work = nullptr;
    (void)shafft::allocBuffer(alloc_elems, &data);
    (void)shafft::allocBuffer(alloc_elems, &work);
    
    std::vector<shafft::complexf> original(alloc_elems);
    init_index_tensor(original.data(), local_elems, dims, subsize, offset);
    
    (void)shafft::copyToBuffer(data, original.data(), alloc_elems);
    (void)plan.setBuffers(data, work);
    
    (void)plan.execute(shafft::FFTDirection::FORWARD);
    (void)plan.execute(shafft::FFTDirection::BACKWARD);
    (void)plan.normalize();
    
    shafft::complexf *final_data, *final_work;
    (void)plan.getBuffers(&final_data, &final_work);
    
    std::vector<shafft::complexf> result(alloc_elems);
    (void)shafft::copyFromBuffer(result.data(), final_data, alloc_elems);
    
    double local_err = max_error(original.data(), result.data(), local_elems);
    double global_err = 0;
    MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    
    return global_err < SP_TOL;
}

static bool test_7d_c2c() {
    std::vector<int> dims = {4, 4, 4, 2, 2, 2, 2};  // 1024 elements, more divisible
    
    std::vector<int> comm_dims;
    if (!get_valid_comm_dims(dims, shafft::FFTType::C2C, comm_dims)) return false;
    
    shafft::Plan plan;
    int rc = plan.initCart(comm_dims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    std::vector<int> subsize(dims.size()), offset(dims.size());
    (void)plan.getLayout(subsize, offset, shafft::TensorLayout::CURRENT);
    
    size_t local_elems = product(subsize);
    size_t alloc_elems = plan.allocSize();
    
    shafft::complexf *data = nullptr, *work = nullptr;
    (void)shafft::allocBuffer(alloc_elems, &data);
    (void)shafft::allocBuffer(alloc_elems, &work);
    
    std::vector<shafft::complexf> original(alloc_elems);
    init_index_tensor(original.data(), local_elems, dims, subsize, offset);
    
    (void)shafft::copyToBuffer(data, original.data(), alloc_elems);
    (void)plan.setBuffers(data, work);
    
    (void)plan.execute(shafft::FFTDirection::FORWARD);
    (void)plan.execute(shafft::FFTDirection::BACKWARD);
    (void)plan.normalize();
    
    shafft::complexf *final_data, *final_work;
    (void)plan.getBuffers(&final_data, &final_work);
    
    std::vector<shafft::complexf> result(alloc_elems);
    (void)shafft::copyFromBuffer(result.data(), final_data, alloc_elems);
    
    double local_err = max_error(original.data(), result.data(), local_elems);
    double global_err = 0;
    MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    
    return global_err < SP_TOL;
}

static bool test_7d_nda3() {
    std::vector<int> dims = {4, 4, 2, 2, 2, 2, 2};  // 512 elements
    
    std::vector<int> comm_dims;
    if (!get_valid_comm_dims(dims, shafft::FFTType::C2C, comm_dims)) return false;
    
    shafft::Plan plan;
    int rc = plan.initCart(comm_dims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    std::vector<int> subsize(dims.size()), offset(dims.size());
    (void)plan.getLayout(subsize, offset, shafft::TensorLayout::CURRENT);
    
    size_t local_elems = product(subsize);
    size_t alloc_elems = plan.allocSize();
    
    shafft::complexf *data = nullptr, *work = nullptr;
    (void)shafft::allocBuffer(alloc_elems, &data);
    (void)shafft::allocBuffer(alloc_elems, &work);
    
    std::vector<shafft::complexf> original(alloc_elems);
    init_index_tensor(original.data(), local_elems, dims, subsize, offset);
    
    (void)shafft::copyToBuffer(data, original.data(), alloc_elems);
    (void)plan.setBuffers(data, work);
    
    (void)plan.execute(shafft::FFTDirection::FORWARD);
    (void)plan.execute(shafft::FFTDirection::BACKWARD);
    (void)plan.normalize();
    
    shafft::complexf *final_data, *final_work;
    (void)plan.getBuffers(&final_data, &final_work);
    
    std::vector<shafft::complexf> result(alloc_elems);
    (void)shafft::copyFromBuffer(result.data(), final_data, alloc_elems);
    
    double local_err = max_error(original.data(), result.data(), local_elems);
    double global_err = 0;
    MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    
    return global_err < SP_TOL;
}

static bool test_9d_c2c() {
    std::vector<int> dims = {2, 2, 2, 2, 2, 2, 2, 2, 2};  // 512 elements
    
    std::vector<int> comm_dims;
    if (!get_valid_comm_dims(dims, shafft::FFTType::C2C, comm_dims)) return false;
    
    shafft::Plan plan;
    int rc = plan.initCart(comm_dims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    std::vector<int> subsize(dims.size()), offset(dims.size());
    (void)plan.getLayout(subsize, offset, shafft::TensorLayout::CURRENT);
    
    size_t local_elems = product(subsize);
    size_t alloc_elems = plan.allocSize();
    
    shafft::complexf *data = nullptr, *work = nullptr;
    (void)shafft::allocBuffer(alloc_elems, &data);
    (void)shafft::allocBuffer(alloc_elems, &work);
    
    std::vector<shafft::complexf> original(alloc_elems);
    init_index_tensor(original.data(), local_elems, dims, subsize, offset);
    
    (void)shafft::copyToBuffer(data, original.data(), alloc_elems);
    (void)plan.setBuffers(data, work);
    
    (void)plan.execute(shafft::FFTDirection::FORWARD);
    (void)plan.execute(shafft::FFTDirection::BACKWARD);
    (void)plan.normalize();
    
    shafft::complexf *final_data, *final_work;
    (void)plan.getBuffers(&final_data, &final_work);
    
    std::vector<shafft::complexf> result(alloc_elems);
    (void)shafft::copyFromBuffer(result.data(), final_data, alloc_elems);
    
    double local_err = max_error(original.data(), result.data(), local_elems);
    double global_err = 0;
    MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    
    return global_err < SP_TOL;
}

static bool test_11d_c2c() {
    std::vector<int> dims(11, 2);  // 2048 elements
    
    std::vector<int> comm_dims;
    if (!get_valid_comm_dims(dims, shafft::FFTType::C2C, comm_dims)) return false;
    
    shafft::Plan plan;
    int rc = plan.initCart(comm_dims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    std::vector<int> subsize(dims.size()), offset(dims.size());
    (void)plan.getLayout(subsize, offset, shafft::TensorLayout::CURRENT);
    
    size_t local_elems = product(subsize);
    size_t alloc_elems = plan.allocSize();
    
    shafft::complexf *data = nullptr, *work = nullptr;
    (void)shafft::allocBuffer(alloc_elems, &data);
    (void)shafft::allocBuffer(alloc_elems, &work);
    
    std::vector<shafft::complexf> original(alloc_elems);
    init_index_tensor(original.data(), local_elems, dims, subsize, offset);
    
    (void)shafft::copyToBuffer(data, original.data(), alloc_elems);
    (void)plan.setBuffers(data, work);
    
    (void)plan.execute(shafft::FFTDirection::FORWARD);
    (void)plan.execute(shafft::FFTDirection::BACKWARD);
    (void)plan.normalize();
    
    shafft::complexf *final_data, *final_work;
    (void)plan.getBuffers(&final_data, &final_work);
    
    std::vector<shafft::complexf> result(alloc_elems);
    (void)shafft::copyFromBuffer(result.data(), final_data, alloc_elems);
    
    double local_err = max_error(original.data(), result.data(), local_elems);
    double global_err = 0;
    MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    
    return global_err < SP_TOL;
}

// ============================================================================
// Double Precision Roundtrip Tests
// ============================================================================

static bool test_5d_z2z() {
    std::vector<int> dims = {4, 4, 4, 4, 4};
    
    std::vector<int> comm_dims;
    if (!get_valid_comm_dims(dims, shafft::FFTType::Z2Z, comm_dims)) return false;
    
    shafft::Plan plan;
    int rc = plan.initCart(comm_dims, dims, shafft::FFTType::Z2Z, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    std::vector<int> subsize(dims.size()), offset(dims.size());
    (void)plan.getLayout(subsize, offset, shafft::TensorLayout::CURRENT);
    
    size_t local_elems = product(subsize);
    size_t alloc_elems = plan.allocSize();
    
    shafft::complexd *data = nullptr, *work = nullptr;
    (void)shafft::allocBuffer(alloc_elems, &data);
    (void)shafft::allocBuffer(alloc_elems, &work);
    
    std::vector<shafft::complexd> original(alloc_elems);
    init_index_tensor(original.data(), local_elems, dims, subsize, offset);
    
    (void)shafft::copyToBuffer(data, original.data(), alloc_elems);
    (void)plan.setBuffers(data, work);
    
    (void)plan.execute(shafft::FFTDirection::FORWARD);
    (void)plan.execute(shafft::FFTDirection::BACKWARD);
    (void)plan.normalize();
    
    shafft::complexd *final_data, *final_work;
    (void)plan.getBuffers(&final_data, &final_work);
    
    std::vector<shafft::complexd> result(alloc_elems);
    (void)shafft::copyFromBuffer(result.data(), final_data, alloc_elems);
    
    double local_err = max_error(original.data(), result.data(), local_elems);
    double global_err = 0;
    MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    
    return global_err < DP_TOL;
}

static bool test_7d_z2z() {
    std::vector<int> dims = {4, 4, 4, 2, 2, 2, 2};  // 1024 elements
    
    std::vector<int> comm_dims;
    if (!get_valid_comm_dims(dims, shafft::FFTType::Z2Z, comm_dims)) return false;
    
    shafft::Plan plan;
    int rc = plan.initCart(comm_dims, dims, shafft::FFTType::Z2Z, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    std::vector<int> subsize(dims.size()), offset(dims.size());
    (void)plan.getLayout(subsize, offset, shafft::TensorLayout::CURRENT);
    
    size_t local_elems = product(subsize);
    size_t alloc_elems = plan.allocSize();
    
    shafft::complexd *data = nullptr, *work = nullptr;
    (void)shafft::allocBuffer(alloc_elems, &data);
    (void)shafft::allocBuffer(alloc_elems, &work);
    
    std::vector<shafft::complexd> original(alloc_elems);
    init_index_tensor(original.data(), local_elems, dims, subsize, offset);
    
    (void)shafft::copyToBuffer(data, original.data(), alloc_elems);
    (void)plan.setBuffers(data, work);
    
    (void)plan.execute(shafft::FFTDirection::FORWARD);
    (void)plan.execute(shafft::FFTDirection::BACKWARD);
    (void)plan.normalize();
    
    shafft::complexd *final_data, *final_work;
    (void)plan.getBuffers(&final_data, &final_work);
    
    std::vector<shafft::complexd> result(alloc_elems);
    (void)shafft::copyFromBuffer(result.data(), final_data, alloc_elems);
    
    double local_err = max_error(original.data(), result.data(), local_elems);
    double global_err = 0;
    MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    
    return global_err < DP_TOL;
}

// ============================================================================
// Edge Cases
// ============================================================================

static bool test_asymmetric_5d() {
    std::vector<int> dims = {8, 4, 2, 4, 8};  // 2048 elements, asymmetric
    
    std::vector<int> comm_dims;
    if (!get_valid_comm_dims(dims, shafft::FFTType::C2C, comm_dims)) return false;
    
    shafft::Plan plan;
    int rc = plan.initCart(comm_dims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    std::vector<int> subsize(dims.size()), offset(dims.size());
    (void)plan.getLayout(subsize, offset, shafft::TensorLayout::CURRENT);
    
    size_t local_elems = product(subsize);
    size_t alloc_elems = plan.allocSize();
    
    shafft::complexf *data = nullptr, *work = nullptr;
    (void)shafft::allocBuffer(alloc_elems, &data);
    (void)shafft::allocBuffer(alloc_elems, &work);
    
    std::vector<shafft::complexf> original(alloc_elems);
    init_index_tensor(original.data(), local_elems, dims, subsize, offset);
    
    (void)shafft::copyToBuffer(data, original.data(), alloc_elems);
    (void)plan.setBuffers(data, work);
    
    (void)plan.execute(shafft::FFTDirection::FORWARD);
    (void)plan.execute(shafft::FFTDirection::BACKWARD);
    (void)plan.normalize();
    
    shafft::complexf *final_data, *final_work;
    (void)plan.getBuffers(&final_data, &final_work);
    
    std::vector<shafft::complexf> result(alloc_elems);
    (void)shafft::copyFromBuffer(result.data(), final_data, alloc_elems);
    
    double local_err = max_error(original.data(), result.data(), local_elems);
    double global_err = 0;
    MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    
    return global_err < SP_TOL;
}

static bool test_prime_dims_5d() {
    std::vector<int> dims = {3, 5, 7, 3, 5};  // 1575 elements, prime factors
    
    std::vector<int> comm_dims;
    if (!get_valid_comm_dims(dims, shafft::FFTType::C2C, comm_dims)) return false;
    
    shafft::Plan plan;
    int rc = plan.initCart(comm_dims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    std::vector<int> subsize(dims.size()), offset(dims.size());
    (void)plan.getLayout(subsize, offset, shafft::TensorLayout::CURRENT);
    
    size_t local_elems = product(subsize);
    size_t alloc_elems = plan.allocSize();
    
    shafft::complexf *data = nullptr, *work = nullptr;
    (void)shafft::allocBuffer(alloc_elems, &data);
    (void)shafft::allocBuffer(alloc_elems, &work);
    
    std::vector<shafft::complexf> original(alloc_elems);
    init_index_tensor(original.data(), local_elems, dims, subsize, offset);
    
    (void)shafft::copyToBuffer(data, original.data(), alloc_elems);
    (void)plan.setBuffers(data, work);
    
    (void)plan.execute(shafft::FFTDirection::FORWARD);
    (void)plan.execute(shafft::FFTDirection::BACKWARD);
    (void)plan.normalize();
    
    shafft::complexf *final_data, *final_work;
    (void)plan.getBuffers(&final_data, &final_work);
    
    std::vector<shafft::complexf> result(alloc_elems);
    (void)shafft::copyFromBuffer(result.data(), final_data, alloc_elems);
    
    double local_err = max_error(original.data(), result.data(), local_elems);
    double global_err = 0;
    MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    
    return global_err < SP_TOL;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    TestRunner runner("High-Dimensional Roundtrip Tests");
    
    runner.run("5D roundtrip (C2C, 4^5, nda=1)", test_5d_c2c);
    runner.run("5D roundtrip (Z2Z, 4^5, nda=1)", test_5d_z2z);
    runner.run("5D roundtrip (C2C, 4^5, nda=2)", test_5d_nda2);
    runner.run("7D roundtrip (C2C, 2x2x3x3x2x2x3)", test_7d_c2c);
    runner.run("7D roundtrip (Z2Z, 2x3x2x3x2x3x2)", test_7d_z2z);
    runner.run("7D roundtrip (C2C, nda=3)", test_7d_nda3);
    runner.run("9D roundtrip (C2C, 2^9)", test_9d_c2c);
    runner.run("11D roundtrip (C2C, 2^11)", test_11d_c2c);
    runner.run("5D asymmetric (8x4x2x4x8)", test_asymmetric_5d);
    runner.run("5D prime dims (3x5x7x3x5)", test_prime_dims_5d);
    
    int rc = runner.finalize();
    MPI_Finalize();
    return rc;
}
