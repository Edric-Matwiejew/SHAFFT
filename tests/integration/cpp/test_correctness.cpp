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
 * Uses configurationND to determine valid process grid for small tensor
 * dimensions to avoid zero-element rank allocation.
 */

#include "test_utils.hpp"
#include <algorithm>
#include <cmath>
#include <mpi.h>
#include <random>
#include <shafft/shafft.hpp>
#include <vector>

constexpr double PI = 3.14159265358979323846;
constexpr double TWO_PI = 2.0 * PI;
constexpr double DP_TOL = 1e-10;

/**
 * Get valid communication dimensions using configurationND.
 * Pass all zeros to let the library automatically determine
 * the best decomposition for the current world size.
 */
static bool get_valid_comm_dims(const std::vector<size_t>& dims,
                                shafft::FFTType type,
                                std::vector<int>& commDims) {
  int ndim = dims.size();
  commDims.resize(ndim, 0);
  std::vector<size_t> subsize(ndim), offset(ndim);
  int commSize = 0;
  int nda = 0;

  int rc = shafft::configurationND(dims,
                                   type,
                                   commDims,
                                   nda,
                                   subsize,
                                   offset,
                                   commSize,
                                   shafft::DecompositionStrategy::MINIMIZE_NDA,
                                   0,
                                   MPI_COMM_WORLD);
  return (rc == 0);
}

/**
 * Initialize delta function at specified global position
 */
void init_delta(shafft::complexd* data,
                size_t local_size,
                const std::vector<size_t>& dims,
                const std::vector<size_t>& subsize,
                const std::vector<size_t>& offset,
                const std::vector<int>& delta_pos) {

  std::fill(data, data + local_size, shafft::complexd(0, 0));

  // Check if delta position is in our local slab
  bool is_local = true;
  std::vector<size_t> local_coords(dims.size());
  for (size_t d = 0; d < dims.size(); ++d) {
    if (static_cast<size_t>(delta_pos[d]) < offset[d] ||
        static_cast<size_t>(delta_pos[d]) >= offset[d] + subsize[d]) {
      is_local = false;
      break;
    }
    local_coords[d] = static_cast<size_t>(delta_pos[d]) - offset[d];
  }

  if (is_local) {
    // Compute local linear index (row-major)
    std::vector<size_t> local_strides(subsize.size());
    local_strides.back() = 1;
    for (int i = subsize.size() - 2; i >= 0; --i) {
      local_strides[i] = local_strides[i + 1] * subsize[i + 1];
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
void init_plane_wave(shafft::complexd* data,
                     size_t local_size,
                     const std::vector<size_t>& dims,
                     const std::vector<size_t>& subsize,
                     const std::vector<size_t>& offset,
                     const std::vector<int>& kvec) {

  std::vector<size_t> local_strides(subsize.size());
  local_strides.back() = 1;
  for (int i = subsize.size() - 2; i >= 0; --i) {
    local_strides[i] = local_strides[i + 1] * subsize[i + 1];
  }

  std::vector<size_t> coords(dims.size());
  for (size_t lin = 0; lin < local_size; ++lin) {
    // Convert to coordinates
    size_t tmp = lin;
    for (size_t d = 0; d < dims.size(); ++d) {
      coords[d] = (tmp / local_strides[d]) % subsize[d];
    }

    // Compute phase
    double phase = 0;
    for (size_t d = 0; d < dims.size(); ++d) {
      size_t global_coord = offset[d] + coords[d];
      phase += TWO_PI * kvec[d] * static_cast<double>(global_coord) / static_cast<double>(dims[d]);
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
  std::vector<size_t> dims = {8, 8, 8};
  std::vector<int> delta_pos = {0, 0, 0};

  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  std::vector<int> commDims(dims.size(), 1);
  commDims[0] = worldSize;

  shafft::FFTND fft;
  int rc = fft.init(commDims, dims, shafft::FFTType::Z2Z, MPI_COMM_WORLD);
  rc = fft.plan();
  if (rc != 0)
    return false;

  std::vector<size_t> subsize(dims.size()), offset(dims.size());
  SHAFFT_CHECK(fft.getLayout(subsize, offset, shafft::TensorLayout::CURRENT));

  size_t localElems = test::product(subsize);
  size_t alloc_elems = fft.allocSize();

  shafft::complexd *data = nullptr, *work = nullptr;
  SHAFFT_CHECK(shafft::allocBuffer(alloc_elems, &data));
  SHAFFT_CHECK(shafft::allocBuffer(alloc_elems, &work));

  std::vector<shafft::complexd> host_data(alloc_elems);
  init_delta(host_data.data(), localElems, dims, subsize, offset, delta_pos);
  SHAFFT_CHECK(shafft::copyToBuffer(data, host_data.data(), alloc_elems));

  SHAFFT_CHECK(fft.setBuffers(data, work));
  SHAFFT_CHECK(fft.execute(shafft::FFTDirection::FORWARD));

  shafft::complexd *final_data, *final_work;
  SHAFFT_CHECK(fft.getBuffers(&final_data, &final_work));

  // Get final layout
  std::vector<size_t> final_subsize(dims.size()), final_offset(dims.size());
  SHAFFT_CHECK(fft.getLayout(final_subsize, final_offset, shafft::TensorLayout::CURRENT));
  size_t final_local_elems = test::product(final_subsize);

  std::vector<shafft::complexd> result(alloc_elems);
  SHAFFT_CHECK(shafft::copyFromBuffer(result.data(), final_data, alloc_elems));

  // FFT of delta at origin should be constant 1.0 everywhere
  double max_dev = 0;
  for (size_t i = 0; i < final_local_elems; ++i) {
    double mag =
        std::sqrt(result[i].real() * result[i].real() + result[i].imag() * result[i].imag());
    double dev = std::fabs(mag - 1.0);
    max_dev = std::max(max_dev, dev);
  }

  double global_max_dev;
  MPI_Allreduce(&max_dev, &global_max_dev, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  (void)shafft::freeBuffer(data);
  (void)shafft::freeBuffer(work);

  return global_max_dev < DP_TOL;
}

// ============================================================================
// Test: Plane wave -> delta spike
// ============================================================================
static bool test_plane_wave_to_delta() {
  std::vector<size_t> dims = {8, 8, 8};
  std::vector<int> kvec = {2, 3, 1};

  size_t N_total = 1;
  for (size_t d : dims)
    N_total *= d;

  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  std::vector<int> commDims(dims.size(), 1);
  commDims[0] = worldSize;

  shafft::FFTND fft;
  int rc = fft.init(commDims, dims, shafft::FFTType::Z2Z, MPI_COMM_WORLD);
  rc = fft.plan();
  if (rc != 0)
    return false;

  std::vector<size_t> subsize(dims.size()), offset(dims.size());
  SHAFFT_CHECK(fft.getLayout(subsize, offset, shafft::TensorLayout::CURRENT));

  size_t localElems = test::product(subsize);
  size_t alloc_elems = fft.allocSize();

  shafft::complexd *data = nullptr, *work = nullptr;
  SHAFFT_CHECK(shafft::allocBuffer(alloc_elems, &data));
  SHAFFT_CHECK(shafft::allocBuffer(alloc_elems, &work));

  std::vector<shafft::complexd> host_data(alloc_elems);
  init_plane_wave(host_data.data(), localElems, dims, subsize, offset, kvec);
  SHAFFT_CHECK(shafft::copyToBuffer(data, host_data.data(), alloc_elems));

  SHAFFT_CHECK(fft.setBuffers(data, work));
  SHAFFT_CHECK(fft.execute(shafft::FFTDirection::FORWARD));

  shafft::complexd *final_data, *final_work;
  SHAFFT_CHECK(fft.getBuffers(&final_data, &final_work));

  std::vector<size_t> final_subsize(dims.size()), final_offset(dims.size());
  SHAFFT_CHECK(fft.getLayout(final_subsize, final_offset, shafft::TensorLayout::CURRENT));
  size_t final_local_elems = test::product(final_subsize);

  std::vector<shafft::complexd> result(alloc_elems);
  SHAFFT_CHECK(shafft::copyFromBuffer(result.data(), final_data, alloc_elems));

  // Find local max magnitude
  double local_max = 0;
  for (size_t i = 0; i < final_local_elems; ++i) {
    double mag =
        std::sqrt(result[i].real() * result[i].real() + result[i].imag() * result[i].imag());
    local_max = std::max(local_max, mag);
  }

  double global_max;
  MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  (void)shafft::freeBuffer(data);
  (void)shafft::freeBuffer(work);

  // Peak should be approximately N_total
  double expected_peak = static_cast<double>(N_total);
  double peak_error = std::fabs(global_max - expected_peak) / expected_peak;

  return peak_error < DP_TOL;
}

// ============================================================================
// Test: Constant input -> DC spike
// ============================================================================
static bool test_constant_to_dc() {
  std::vector<size_t> dims = {8, 8, 8};
  shafft::complexd constant_val(1.5, -0.5);

  size_t N_total = 1;
  for (size_t d : dims)
    N_total *= d;

  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  std::vector<int> commDims(dims.size(), 1);
  commDims[0] = worldSize;

  shafft::FFTND fft;
  int rc = fft.init(commDims, dims, shafft::FFTType::Z2Z, MPI_COMM_WORLD);
  rc = fft.plan();
  if (rc != 0)
    return false;

  std::vector<size_t> subsize(dims.size()), offset(dims.size());
  SHAFFT_CHECK(fft.getLayout(subsize, offset, shafft::TensorLayout::CURRENT));

  size_t alloc_elems = fft.allocSize();

  shafft::complexd *data = nullptr, *work = nullptr;
  SHAFFT_CHECK(shafft::allocBuffer(alloc_elems, &data));
  SHAFFT_CHECK(shafft::allocBuffer(alloc_elems, &work));

  std::vector<shafft::complexd> host_data(alloc_elems, constant_val);
  SHAFFT_CHECK(shafft::copyToBuffer(data, host_data.data(), alloc_elems));

  SHAFFT_CHECK(fft.setBuffers(data, work));
  SHAFFT_CHECK(fft.execute(shafft::FFTDirection::FORWARD));

  shafft::complexd *final_data, *final_work;
  SHAFFT_CHECK(fft.getBuffers(&final_data, &final_work));

  std::vector<size_t> final_subsize(dims.size()), final_offset(dims.size());
  SHAFFT_CHECK(fft.getLayout(final_subsize, final_offset, shafft::TensorLayout::CURRENT));
  size_t final_local_elems = test::product(final_subsize);

  std::vector<shafft::complexd> result(alloc_elems);
  SHAFFT_CHECK(shafft::copyFromBuffer(result.data(), final_data, alloc_elems));

  // Find max magnitude (should be DC)
  double local_max = 0;
  for (size_t i = 0; i < final_local_elems; ++i) {
    double mag =
        std::sqrt(result[i].real() * result[i].real() + result[i].imag() * result[i].imag());
    local_max = std::max(local_max, mag);
  }

  double global_max;
  MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  (void)shafft::freeBuffer(data);
  (void)shafft::freeBuffer(work);

  // DC should be constant_val * N_total
  double expected_mag = std::sqrt(constant_val.real() * constant_val.real() +
                                  constant_val.imag() * constant_val.imag()) *
                        N_total;
  double error = std::fabs(global_max - expected_mag) / expected_mag;

  return error < DP_TOL;
}

// ============================================================================
// Test: Parseval's theorem
// ============================================================================
static bool test_parseval() {
  std::vector<size_t> dims = {16, 16, 8};

  size_t N_total = 1;
  for (size_t d : dims)
    N_total *= d;

  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  std::vector<int> commDims(dims.size(), 1);
  commDims[0] = worldSize;

  shafft::FFTND fft;
  int rc = fft.init(commDims, dims, shafft::FFTType::Z2Z, MPI_COMM_WORLD);
  rc = fft.plan();
  if (rc != 0)
    return false;

  std::vector<size_t> subsize(dims.size()), offset(dims.size());
  SHAFFT_CHECK(fft.getLayout(subsize, offset, shafft::TensorLayout::CURRENT));

  size_t localElems = test::product(subsize);
  size_t alloc_elems = fft.allocSize();

  shafft::complexd *data = nullptr, *work = nullptr;
  SHAFFT_CHECK(shafft::allocBuffer(alloc_elems, &data));
  SHAFFT_CHECK(shafft::allocBuffer(alloc_elems, &work));

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Initialize with random data
  std::vector<shafft::complexd> host_data(alloc_elems);
  std::mt19937_64 rng(42 + rank);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (size_t i = 0; i < localElems; ++i) {
    host_data[i] = shafft::complexd(dist(rng), dist(rng));
  }

  // Compute input energy
  double local_input_energy = local_energy(host_data.data(), localElems);
  double global_input_energy;
  MPI_Allreduce(&local_input_energy, &global_input_energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  SHAFFT_CHECK(shafft::copyToBuffer(data, host_data.data(), alloc_elems));
  SHAFFT_CHECK(fft.setBuffers(data, work));
  SHAFFT_CHECK(fft.execute(shafft::FFTDirection::FORWARD));

  shafft::complexd *final_data, *final_work;
  SHAFFT_CHECK(fft.getBuffers(&final_data, &final_work));

  std::vector<size_t> final_subsize(dims.size()), final_offset(dims.size());
  SHAFFT_CHECK(fft.getLayout(final_subsize, final_offset, shafft::TensorLayout::CURRENT));
  size_t final_local_elems = test::product(final_subsize);

  std::vector<shafft::complexd> result(alloc_elems);
  SHAFFT_CHECK(shafft::copyFromBuffer(result.data(), final_data, alloc_elems));

  // Compute output energy
  double local_output_energy = local_energy(result.data(), final_local_elems);
  double global_output_energy;
  MPI_Allreduce(
      &local_output_energy, &global_output_energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  (void)shafft::freeBuffer(data);
  (void)shafft::freeBuffer(work);

  // Parseval: sum|f|^2 = (1/N) sum|F|^2
  double expected_output = global_input_energy * N_total;
  double rel_error = std::fabs(global_output_energy - expected_output) / expected_output;

  return rel_error < DP_TOL;
}

// ============================================================================
// Test: High-dimensional plane wave (5D)
// ============================================================================
static bool test_5d_plane_wave() {
  std::vector<size_t> dims = {4, 4, 4, 4, 4};
  std::vector<int> kvec = {1, 2, 0, 3, 1};

  size_t N_total = 1;
  for (size_t d : dims)
    N_total *= d;

  std::vector<int> commDims;
  if (!get_valid_comm_dims(dims, shafft::FFTType::Z2Z, commDims))
    return false;

  shafft::FFTND fft;
  int rc = fft.init(commDims, dims, shafft::FFTType::Z2Z, MPI_COMM_WORLD);
  rc = fft.plan();
  if (rc != 0)
    return false;

  std::vector<size_t> subsize(dims.size()), offset(dims.size());
  SHAFFT_CHECK(fft.getLayout(subsize, offset, shafft::TensorLayout::CURRENT));

  size_t localElems = test::product(subsize);
  size_t alloc_elems = fft.allocSize();

  shafft::complexd *data = nullptr, *work = nullptr;
  SHAFFT_CHECK(shafft::allocBuffer(alloc_elems, &data));
  SHAFFT_CHECK(shafft::allocBuffer(alloc_elems, &work));

  std::vector<shafft::complexd> host_data(alloc_elems);
  init_plane_wave(host_data.data(), localElems, dims, subsize, offset, kvec);
  SHAFFT_CHECK(shafft::copyToBuffer(data, host_data.data(), alloc_elems));

  SHAFFT_CHECK(fft.setBuffers(data, work));
  SHAFFT_CHECK(fft.execute(shafft::FFTDirection::FORWARD));

  shafft::complexd *final_data, *final_work;
  SHAFFT_CHECK(fft.getBuffers(&final_data, &final_work));

  std::vector<size_t> final_subsize(dims.size()), final_offset(dims.size());
  SHAFFT_CHECK(fft.getLayout(final_subsize, final_offset, shafft::TensorLayout::CURRENT));
  size_t final_local_elems = test::product(final_subsize);

  std::vector<shafft::complexd> result(alloc_elems);
  SHAFFT_CHECK(shafft::copyFromBuffer(result.data(), final_data, alloc_elems));

  // Find max
  double local_max = 0;
  for (size_t i = 0; i < final_local_elems; ++i) {
    double mag =
        std::sqrt(result[i].real() * result[i].real() + result[i].imag() * result[i].imag());
    local_max = std::max(local_max, mag);
  }

  double global_max;
  MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  (void)shafft::freeBuffer(data);
  (void)shafft::freeBuffer(work);

  double expected_peak = static_cast<double>(N_total);
  double peak_error = std::fabs(global_max - expected_peak) / expected_peak;

  return peak_error < DP_TOL;
}

// ============================================================================
// Test: 7D delta - use dimensions divisible by common rank counts
// ============================================================================
static bool test_7d_delta() {
  std::vector<size_t> dims = {4, 4, 4, 4, 2, 2, 2}; // 2048 elements, more divisible
  std::vector<int> delta_pos(dims.size(), 0);

  std::vector<int> commDims;
  if (!get_valid_comm_dims(dims, shafft::FFTType::Z2Z, commDims))
    return false;

  shafft::FFTND fft;
  int rc = fft.init(commDims, dims, shafft::FFTType::Z2Z, MPI_COMM_WORLD);
  rc = fft.plan();
  if (rc != 0)
    return false;

  std::vector<size_t> subsize(dims.size()), offset(dims.size());
  SHAFFT_CHECK(fft.getLayout(subsize, offset, shafft::TensorLayout::CURRENT));

  size_t localElems = test::product(subsize);
  size_t alloc_elems = fft.allocSize();

  shafft::complexd *data = nullptr, *work = nullptr;
  SHAFFT_CHECK(shafft::allocBuffer(alloc_elems, &data));
  SHAFFT_CHECK(shafft::allocBuffer(alloc_elems, &work));

  std::vector<shafft::complexd> host_data(alloc_elems);
  init_delta(host_data.data(), localElems, dims, subsize, offset, delta_pos);
  SHAFFT_CHECK(shafft::copyToBuffer(data, host_data.data(), alloc_elems));

  SHAFFT_CHECK(fft.setBuffers(data, work));
  SHAFFT_CHECK(fft.execute(shafft::FFTDirection::FORWARD));

  shafft::complexd *final_data, *final_work;
  SHAFFT_CHECK(fft.getBuffers(&final_data, &final_work));

  std::vector<size_t> final_subsize(dims.size()), final_offset(dims.size());
  SHAFFT_CHECK(fft.getLayout(final_subsize, final_offset, shafft::TensorLayout::CURRENT));
  size_t final_local_elems = test::product(final_subsize);

  std::vector<shafft::complexd> result(alloc_elems);
  SHAFFT_CHECK(shafft::copyFromBuffer(result.data(), final_data, alloc_elems));

  // All values should have magnitude 1
  double max_dev = 0;
  for (size_t i = 0; i < final_local_elems; ++i) {
    double mag =
        std::sqrt(result[i].real() * result[i].real() + result[i].imag() * result[i].imag());
    double dev = std::fabs(mag - 1.0);
    max_dev = std::max(max_dev, dev);
  }

  double global_max_dev;
  MPI_Allreduce(&max_dev, &global_max_dev, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  (void)shafft::freeBuffer(data);
  (void)shafft::freeBuffer(work);

  return global_max_dev < DP_TOL;
}

int main(int argc, char** argv) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::fprintf(stderr, "MPI_Init failed\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  test::TestRunnerMPI runner("Mathematical Correctness Tests");

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
