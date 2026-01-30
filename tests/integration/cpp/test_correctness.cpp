/**
 * @file test_correctness.cpp
 * @brief Mathematical correctness tests for FFT transforms
 * 
 * These tests verify that the FFT produces mathematically correct results,
 * not just roundtrip consistency. Based on known DFT properties:
 * 
 * 1. Delta (impulse) test: delta(x) -> constant in frequency domain
 * 2. Plane wave test: exp(2*pi*i * k.x/N) -> delta(k) in frequency domain
 * 3. Constant input test: constant -> delta(0) in frequency domain
 * 4. Parseval's theorem: sum|f(x)|^2 = (1/N) sum|F(k)|^2
 * 
 * Uses configurationCart to determine valid process grid for small tensor
 * dimensions to avoid zero-element rank allocation.
 */

#include "test_utils.hpp"
#include <shafft/shafft.hpp>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

using namespace test;

constexpr double PI = 3.14159265358979323846;
constexpr double TWO_PI = 2.0 * PI;
constexpr double DP_TOL = 1e-10;

/**
 * Get valid communication dimensions using configurationCart.
 * Pass all zeros to let the library automatically determine
 * the best decomposition for the current world size.
 */
static bool get_valid_comm_dims(const std::vector<int>& dims,
                                 shafft::FFTType type,
                                 std::vector<int>& comm_dims) {
    int ndim = dims.size();
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

/**
 * Initialize delta function at specified global position
 */
void init_delta(shafft::complexd* data, size_t local_size,
                const std::vector<int>& dims,
                const std::vector<int>& subsize,
                const std::vector<int>& offset,
                const std::vector<int>& delta_pos) {
    
    std::fill(data, data + local_size, shafft::complexd(0, 0));
    
    // Check if delta position is in our local slab
    bool is_local = true;
    std::vector<int> local_coords(dims.size());
    for (size_t d = 0; d < dims.size(); ++d) {
        if (delta_pos[d] < offset[d] || delta_pos[d] >= offset[d] + subsize[d]) {
            is_local = false;
            break;
        }
        local_coords[d] = delta_pos[d] - offset[d];
    }
    
    if (is_local) {
        // Compute local linear index (row-major)
        std::vector<size_t> local_strides(subsize.size());
        local_strides.back() = 1;
        for (int i = subsize.size() - 2; i >= 0; --i) {
            local_strides[i] = local_strides[i+1] * subsize[i+1];
        }
        
        size_t lin = 0;
        for (size_t d = 0; d < dims.size(); ++d) {
            lin += local_coords[d] * local_strides[d];
        }
        data[lin] = shafft::complexd(1, 0);
    }
}

/**
 * Initialize plane wave: exp(2*pi*i * k.x/N)
 */
void init_plane_wave(shafft::complexd* data, size_t local_size,
                     const std::vector<int>& dims,
                     const std::vector<int>& subsize,
                     const std::vector<int>& offset,
                     const std::vector<int>& kvec) {
    
    std::vector<size_t> local_strides(subsize.size());
    local_strides.back() = 1;
    for (int i = subsize.size() - 2; i >= 0; --i) {
        local_strides[i] = local_strides[i+1] * subsize[i+1];
    }
    
    std::vector<int> coords(dims.size());
    for (size_t lin = 0; lin < local_size; ++lin) {
        // Convert to coordinates
        size_t tmp = lin;
        for (size_t d = 0; d < dims.size(); ++d) {
            coords[d] = (tmp / local_strides[d]) % subsize[d];
        }
        
        // Compute phase
        double phase = 0;
        for (size_t d = 0; d < dims.size(); ++d) {
            int global_coord = offset[d] + coords[d];
            phase += TWO_PI * kvec[d] * global_coord / dims[d];
        }
        
        data[lin] = shafft::complexd(std::cos(phase), std::sin(phase));
    }
}

/**
 * Compute local sum of squared magnitudes
 */
double local_energy(const shafft::complexd* data, size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += data[i].real() * data[i].real() + data[i].imag() * data[i].imag();
    }
    return sum;
}

// ============================================================================
// Test: Delta function -> uniform spectrum
// ============================================================================
static bool test_delta_uniform_spectrum() {
    std::vector<int> dims = {8, 8, 8};
    std::vector<int> delta_pos = {0, 0, 0};
    int nda = 1;
    
    shafft::Plan plan;
    int rc = plan.init(nda, dims, shafft::FFTType::Z2Z, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    std::vector<int> subsize(dims.size()), offset(dims.size());
    plan.getLayout(subsize, offset, shafft::TensorLayout::CURRENT);
    
    size_t local_elems = product(subsize);
    size_t alloc_elems = plan.allocSize();
    
    shafft::complexd *data = nullptr, *work = nullptr;
    shafft::allocBuffer(alloc_elems, &data);
    shafft::allocBuffer(alloc_elems, &work);
    
    std::vector<shafft::complexd> host_data(alloc_elems);
    init_delta(host_data.data(), local_elems, dims, subsize, offset, delta_pos);
    shafft::copyToBuffer(data, host_data.data(), alloc_elems);
    
    plan.setBuffers(data, work);
    plan.execute(shafft::FFTDirection::FORWARD);
    
    shafft::complexd *final_data, *final_work;
    plan.getBuffers(&final_data, &final_work);
    
    // Get final layout
    std::vector<int> final_subsize(dims.size()), final_offset(dims.size());
    plan.getLayout(final_subsize, final_offset, shafft::TensorLayout::CURRENT);
    size_t final_local_elems = product(final_subsize);
    
    std::vector<shafft::complexd> result(alloc_elems);
    shafft::copyFromBuffer(result.data(), final_data, alloc_elems);
    
    // FFT of delta at origin should be constant 1.0 everywhere
    double max_dev = 0;
    for (size_t i = 0; i < final_local_elems; ++i) {
        double mag = std::sqrt(result[i].real() * result[i].real() + 
                               result[i].imag() * result[i].imag());
        double dev = std::fabs(mag - 1.0);
        max_dev = std::max(max_dev, dev);
    }
    
    double global_max_dev;
    MPI_Allreduce(&max_dev, &global_max_dev, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    shafft::freeBuffer(data);
    shafft::freeBuffer(work);
    
    return global_max_dev < DP_TOL;
}

// ============================================================================
// Test: Plane wave -> delta spike
// ============================================================================
static bool test_plane_wave_to_delta() {
    std::vector<int> dims = {8, 8, 8};
    std::vector<int> kvec = {2, 3, 1};
    int nda = 1;
    
    size_t N_total = 1;
    for (int d : dims) N_total *= d;
    
    shafft::Plan plan;
    int rc = plan.init(nda, dims, shafft::FFTType::Z2Z, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    std::vector<int> subsize(dims.size()), offset(dims.size());
    plan.getLayout(subsize, offset, shafft::TensorLayout::CURRENT);
    
    size_t local_elems = product(subsize);
    size_t alloc_elems = plan.allocSize();
    
    shafft::complexd *data = nullptr, *work = nullptr;
    shafft::allocBuffer(alloc_elems, &data);
    shafft::allocBuffer(alloc_elems, &work);
    
    std::vector<shafft::complexd> host_data(alloc_elems);
    init_plane_wave(host_data.data(), local_elems, dims, subsize, offset, kvec);
    shafft::copyToBuffer(data, host_data.data(), alloc_elems);
    
    plan.setBuffers(data, work);
    plan.execute(shafft::FFTDirection::FORWARD);
    
    shafft::complexd *final_data, *final_work;
    plan.getBuffers(&final_data, &final_work);
    
    std::vector<int> final_subsize(dims.size()), final_offset(dims.size());
    plan.getLayout(final_subsize, final_offset, shafft::TensorLayout::CURRENT);
    size_t final_local_elems = product(final_subsize);
    
    std::vector<shafft::complexd> result(alloc_elems);
    shafft::copyFromBuffer(result.data(), final_data, alloc_elems);
    
    // Find local max magnitude
    double local_max = 0;
    for (size_t i = 0; i < final_local_elems; ++i) {
        double mag = std::sqrt(result[i].real() * result[i].real() + 
                               result[i].imag() * result[i].imag());
        local_max = std::max(local_max, mag);
    }
    
    double global_max;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    shafft::freeBuffer(data);
    shafft::freeBuffer(work);
    
    // Peak should be approximately N_total
    double expected_peak = static_cast<double>(N_total);
    double peak_error = std::fabs(global_max - expected_peak) / expected_peak;
    
    return peak_error < DP_TOL;
}

// ============================================================================
// Test: Constant input -> DC spike
// ============================================================================
static bool test_constant_to_dc() {
    std::vector<int> dims = {8, 8, 8};
    shafft::complexd constant_val(1.5, -0.5);
    int nda = 1;
    
    size_t N_total = 1;
    for (int d : dims) N_total *= d;
    
    shafft::Plan plan;
    int rc = plan.init(nda, dims, shafft::FFTType::Z2Z, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    std::vector<int> subsize(dims.size()), offset(dims.size());
    plan.getLayout(subsize, offset, shafft::TensorLayout::CURRENT);
    
    size_t local_elems = product(subsize);
    size_t alloc_elems = plan.allocSize();
    
    shafft::complexd *data = nullptr, *work = nullptr;
    shafft::allocBuffer(alloc_elems, &data);
    shafft::allocBuffer(alloc_elems, &work);
    
    std::vector<shafft::complexd> host_data(alloc_elems, constant_val);
    shafft::copyToBuffer(data, host_data.data(), alloc_elems);
    
    plan.setBuffers(data, work);
    plan.execute(shafft::FFTDirection::FORWARD);
    
    shafft::complexd *final_data, *final_work;
    plan.getBuffers(&final_data, &final_work);
    
    std::vector<int> final_subsize(dims.size()), final_offset(dims.size());
    plan.getLayout(final_subsize, final_offset, shafft::TensorLayout::CURRENT);
    size_t final_local_elems = product(final_subsize);
    
    std::vector<shafft::complexd> result(alloc_elems);
    shafft::copyFromBuffer(result.data(), final_data, alloc_elems);
    
    // Find max magnitude (should be DC)
    double local_max = 0;
    for (size_t i = 0; i < final_local_elems; ++i) {
        double mag = std::sqrt(result[i].real() * result[i].real() + 
                               result[i].imag() * result[i].imag());
        local_max = std::max(local_max, mag);
    }
    
    double global_max;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    shafft::freeBuffer(data);
    shafft::freeBuffer(work);
    
    // DC should be constant_val * N_total
    double expected_mag = std::sqrt(constant_val.real() * constant_val.real() + 
                                    constant_val.imag() * constant_val.imag()) * N_total;
    double error = std::fabs(global_max - expected_mag) / expected_mag;
    
    return error < DP_TOL;
}

// ============================================================================
// Test: Parseval's theorem
// ============================================================================
static bool test_parseval() {
    std::vector<int> dims = {16, 16, 8};
    int nda = 1;
    
    size_t N_total = 1;
    for (int d : dims) N_total *= d;
    
    shafft::Plan plan;
    int rc = plan.init(nda, dims, shafft::FFTType::Z2Z, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    std::vector<int> subsize(dims.size()), offset(dims.size());
    plan.getLayout(subsize, offset, shafft::TensorLayout::CURRENT);
    
    size_t local_elems = product(subsize);
    size_t alloc_elems = plan.allocSize();
    
    shafft::complexd *data = nullptr, *work = nullptr;
    shafft::allocBuffer(alloc_elems, &data);
    shafft::allocBuffer(alloc_elems, &work);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Initialize with random data
    std::vector<shafft::complexd> host_data(alloc_elems);
    std::mt19937_64 rng(42 + rank);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < local_elems; ++i) {
        host_data[i] = shafft::complexd(dist(rng), dist(rng));
    }
    
    // Compute input energy
    double local_input_energy = local_energy(host_data.data(), local_elems);
    double global_input_energy;
    MPI_Allreduce(&local_input_energy, &global_input_energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    shafft::copyToBuffer(data, host_data.data(), alloc_elems);
    plan.setBuffers(data, work);
    plan.execute(shafft::FFTDirection::FORWARD);
    
    shafft::complexd *final_data, *final_work;
    plan.getBuffers(&final_data, &final_work);
    
    std::vector<int> final_subsize(dims.size()), final_offset(dims.size());
    plan.getLayout(final_subsize, final_offset, shafft::TensorLayout::CURRENT);
    size_t final_local_elems = product(final_subsize);
    
    std::vector<shafft::complexd> result(alloc_elems);
    shafft::copyFromBuffer(result.data(), final_data, alloc_elems);
    
    // Compute output energy
    double local_output_energy = local_energy(result.data(), final_local_elems);
    double global_output_energy;
    MPI_Allreduce(&local_output_energy, &global_output_energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    shafft::freeBuffer(data);
    shafft::freeBuffer(work);
    
    // Parseval: sum|f|^2 = (1/N) sum|F|^2
    double expected_output = global_input_energy * N_total;
    double rel_error = std::fabs(global_output_energy - expected_output) / expected_output;
    
    return rel_error < DP_TOL;
}

// ============================================================================
// Test: High-dimensional plane wave (5D)
// ============================================================================
static bool test_5d_plane_wave() {
    std::vector<int> dims = {4, 4, 4, 4, 4};
    std::vector<int> kvec = {1, 2, 0, 3, 1};
    
    size_t N_total = 1;
    for (int d : dims) N_total *= d;
    
    std::vector<int> comm_dims;
    if (!get_valid_comm_dims(dims, shafft::FFTType::Z2Z, comm_dims)) return false;
    
    shafft::Plan plan;
    int rc = plan.initCart(comm_dims, dims, shafft::FFTType::Z2Z, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    std::vector<int> subsize(dims.size()), offset(dims.size());
    plan.getLayout(subsize, offset, shafft::TensorLayout::CURRENT);
    
    size_t local_elems = product(subsize);
    size_t alloc_elems = plan.allocSize();
    
    shafft::complexd *data = nullptr, *work = nullptr;
    shafft::allocBuffer(alloc_elems, &data);
    shafft::allocBuffer(alloc_elems, &work);
    
    std::vector<shafft::complexd> host_data(alloc_elems);
    init_plane_wave(host_data.data(), local_elems, dims, subsize, offset, kvec);
    shafft::copyToBuffer(data, host_data.data(), alloc_elems);
    
    plan.setBuffers(data, work);
    plan.execute(shafft::FFTDirection::FORWARD);
    
    shafft::complexd *final_data, *final_work;
    plan.getBuffers(&final_data, &final_work);
    
    std::vector<int> final_subsize(dims.size()), final_offset(dims.size());
    plan.getLayout(final_subsize, final_offset, shafft::TensorLayout::CURRENT);
    size_t final_local_elems = product(final_subsize);
    
    std::vector<shafft::complexd> result(alloc_elems);
    shafft::copyFromBuffer(result.data(), final_data, alloc_elems);
    
    // Find max
    double local_max = 0;
    for (size_t i = 0; i < final_local_elems; ++i) {
        double mag = std::sqrt(result[i].real() * result[i].real() + 
                               result[i].imag() * result[i].imag());
        local_max = std::max(local_max, mag);
    }
    
    double global_max;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    shafft::freeBuffer(data);
    shafft::freeBuffer(work);
    
    double expected_peak = static_cast<double>(N_total);
    double peak_error = std::fabs(global_max - expected_peak) / expected_peak;
    
    return peak_error < DP_TOL;
}

// ============================================================================
// Test: 7D delta - use dimensions divisible by common rank counts
// ============================================================================
static bool test_7d_delta() {
    std::vector<int> dims = {4, 4, 4, 4, 2, 2, 2};  // 2048 elements, more divisible
    std::vector<int> delta_pos(dims.size(), 0);
    
    std::vector<int> comm_dims;
    if (!get_valid_comm_dims(dims, shafft::FFTType::Z2Z, comm_dims)) return false;
    
    shafft::Plan plan;
    int rc = plan.initCart(comm_dims, dims, shafft::FFTType::Z2Z, MPI_COMM_WORLD);
    if (rc != 0) return false;
    
    std::vector<int> subsize(dims.size()), offset(dims.size());
    plan.getLayout(subsize, offset, shafft::TensorLayout::CURRENT);
    
    size_t local_elems = product(subsize);
    size_t alloc_elems = plan.allocSize();
    
    shafft::complexd *data = nullptr, *work = nullptr;
    shafft::allocBuffer(alloc_elems, &data);
    shafft::allocBuffer(alloc_elems, &work);
    
    std::vector<shafft::complexd> host_data(alloc_elems);
    init_delta(host_data.data(), local_elems, dims, subsize, offset, delta_pos);
    shafft::copyToBuffer(data, host_data.data(), alloc_elems);
    
    plan.setBuffers(data, work);
    plan.execute(shafft::FFTDirection::FORWARD);
    
    shafft::complexd *final_data, *final_work;
    plan.getBuffers(&final_data, &final_work);
    
    std::vector<int> final_subsize(dims.size()), final_offset(dims.size());
    plan.getLayout(final_subsize, final_offset, shafft::TensorLayout::CURRENT);
    size_t final_local_elems = product(final_subsize);
    
    std::vector<shafft::complexd> result(alloc_elems);
    shafft::copyFromBuffer(result.data(), final_data, alloc_elems);
    
    // All values should have magnitude 1
    double max_dev = 0;
    for (size_t i = 0; i < final_local_elems; ++i) {
        double mag = std::sqrt(result[i].real() * result[i].real() + 
                               result[i].imag() * result[i].imag());
        double dev = std::fabs(mag - 1.0);
        max_dev = std::max(max_dev, dev);
    }
    
    double global_max_dev;
    MPI_Allreduce(&max_dev, &global_max_dev, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    shafft::freeBuffer(data);
    shafft::freeBuffer(work);
    
    return global_max_dev < DP_TOL;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    TestRunner runner("Mathematical Correctness Tests");
    
    runner.run("Delta(0) -> uniform spectrum", test_delta_uniform_spectrum);
    runner.run("Plane wave -> delta spike", test_plane_wave_to_delta);
    runner.run("Constant -> DC spike", test_constant_to_dc);
    runner.run("Parseval's theorem", test_parseval);
    runner.run("5D plane wave", test_5d_plane_wave);
    runner.run("7D delta", test_7d_delta);
    
    int rc = runner.finalize();
    MPI_Finalize();
    return rc;
}
