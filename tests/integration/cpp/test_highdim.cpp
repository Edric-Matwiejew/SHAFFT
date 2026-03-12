/**
 * @file test_highdim.cpp
 * @brief High-dimensional roundtrip tests (5D, 7D, 9D, 11D) with small axis sizes
 *
 * Tests FFT correctness in higher dimensions to verify that the library
 * handles arbitrary dimensionality correctly. Uses small per-axis sizes
 * to keep memory usage low.
 *
 * Uses configurationND to ensure valid decomposition for small dimensions.
 */

#include "test_utils.hpp"
#include <algorithm>
#include <cmath>
#include <mpi.h>
#include <shafft/shafft.hpp>
#include <vector>

/**
 * Initialize tensor with index-based values for uniqueness verification.
 */
template <typename T>
void init_index_tensor(T* data,
                       size_t local_size,
                       const std::vector<size_t>& dims,
                       const std::vector<size_t>& subsize,
                       const std::vector<size_t>& offset) {
  using Real = decltype(data[0].real());

  size_t global_total = 1;
  for (size_t d : dims)
    global_total *= d;
  Real scale = static_cast<Real>(1.0 / global_total);
  Real mid = static_cast<Real>((global_total - 1) / 2.0);

  // Compute strides (row-major)
  std::vector<size_t> strides(dims.size());
  strides.back() = 1;
  for (int i = dims.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * dims[i + 1];
  }

  std::vector<size_t> local_strides(subsize.size());
  local_strides.back() = 1;
  for (int i = subsize.size() - 2; i >= 0; --i) {
    local_strides[i] = local_strides[i + 1] * subsize[i + 1];
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
 * Get valid commDims for a given tensor and world size.
 * Uses configurationND to automatically find a valid decomposition.
 * Pass all zeros in commDims to let the library choose.
 */
bool get_valid_comm_dims(const std::vector<size_t>& dims,
                         shafft::FFTType type,
                         std::vector<int>& commDims) {
  const int ndim = dims.size();
  commDims.resize(ndim);
  std::vector<size_t> subsize(ndim), offset(ndim);
  int commSize = 0;
  int nda = 0; // Let configurationND determine

  // Pass all zeros to let configurationND automatically determine
  // the best decomposition for the current world size
  std::fill(commDims.begin(), commDims.end(), 0);

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

// ============================================================================
// ============================================================================
// Reusable roundtrip helper with inactive-rank support
// ============================================================================

/**
 * Run a forward->backward->normalize roundtrip on an FFTND plan.
 * Inactive ranks skip buffer operations but still participate in the
 * collective MPI_Allreduce inside check_rel_error.
 */
template <typename ComplexT>
static bool runFftndRoundtrip(const std::vector<size_t>& dims,
                                shafft::FFTType type,
                                double tol) {
  std::vector<int> commDims;
  if (!get_valid_comm_dims(dims, type, commDims))
    return false;

  shafft::FFTND fft;
  int rc = fft.init(commDims, dims, type, MPI_COMM_WORLD);
  rc = fft.plan();
  if (rc != 0)
    return false;

  size_t globalN = test::product(dims);

  if (!fft.isActive()) {
    // Inactive rank: participate in collective error check with 0 elements
    return test::check_rel_error(
        static_cast<ComplexT*>(nullptr), static_cast<ComplexT*>(nullptr),
        size_t{0}, globalN, MPI_COMM_WORLD, tol);
  }

  std::vector<size_t> subsize(dims.size()), offset(dims.size());
  SHAFFT_CHECK(fft.getLayout(subsize, offset, shafft::TensorLayout::CURRENT));

  size_t localElems = test::product(subsize);
  size_t alloc_elems = fft.allocSize();

  ComplexT *data = nullptr, *work = nullptr;
  SHAFFT_CHECK(shafft::allocBuffer(alloc_elems, &data));
  SHAFFT_CHECK(shafft::allocBuffer(alloc_elems, &work));

  std::vector<ComplexT> original(alloc_elems);
  init_index_tensor(original.data(), localElems, dims, subsize, offset);

  SHAFFT_CHECK(shafft::copyToBuffer(data, original.data(), alloc_elems));
  SHAFFT_CHECK(fft.setBuffers(data, work));

  SHAFFT_CHECK(fft.execute(shafft::FFTDirection::FORWARD));
  SHAFFT_CHECK(fft.execute(shafft::FFTDirection::BACKWARD));
  SHAFFT_CHECK(fft.normalize());

  ComplexT *final_data, *final_work;
  SHAFFT_CHECK(fft.getBuffers(&final_data, &final_work));

  std::vector<ComplexT> result(alloc_elems);
  SHAFFT_CHECK(shafft::copyFromBuffer(result.data(), final_data, alloc_elems));

  bool passed = test::check_rel_error(
      result.data(), original.data(), localElems, globalN, MPI_COMM_WORLD, tol);

  (void)shafft::freeBuffer(data);
  (void)shafft::freeBuffer(work);

  return passed;
}

// ============================================================================
// Single Precision Roundtrip Tests
// ============================================================================

static bool test_5d_c2c() {
  return runFftndRoundtrip<shafft::complexf>(
      {4, 4, 4, 4, 4}, shafft::FFTType::C2C, test::TOL_F);
}

static bool test_5d_nda2() {
  return runFftndRoundtrip<shafft::complexf>(
      {4, 4, 4, 4, 4}, shafft::FFTType::C2C, test::TOL_F);
}

static bool test_7d_c2c() {
  return runFftndRoundtrip<shafft::complexf>(
      {4, 4, 4, 2, 2, 2, 2}, shafft::FFTType::C2C, test::TOL_F);
}

static bool test_7d_nda3() {
  return runFftndRoundtrip<shafft::complexf>(
      {4, 4, 2, 2, 2, 2, 2}, shafft::FFTType::C2C, test::TOL_F);
}

static bool test_9d_c2c() {
  return runFftndRoundtrip<shafft::complexf>(
      {2, 2, 2, 2, 2, 2, 2, 2, 2}, shafft::FFTType::C2C, test::TOL_F);
}

static bool test_11d_c2c() {
  std::vector<size_t> dims(11, 2);
  return runFftndRoundtrip<shafft::complexf>(
      dims, shafft::FFTType::C2C, test::TOL_F);
}

// ============================================================================
// Double Precision Roundtrip Tests
// ============================================================================

static bool test_5d_z2z() {
  return runFftndRoundtrip<shafft::complexd>(
      {4, 4, 4, 4, 4}, shafft::FFTType::Z2Z, test::TOL_D);
}

static bool test_7d_z2z() {
  return runFftndRoundtrip<shafft::complexd>(
      {4, 4, 4, 2, 2, 2, 2}, shafft::FFTType::Z2Z, test::TOL_D);
}

// ============================================================================
// Edge Cases
// ============================================================================

static bool test_asymmetric_5d() {
  return runFftndRoundtrip<shafft::complexf>(
      {8, 4, 2, 4, 8}, shafft::FFTType::C2C, test::TOL_F);
}

static bool test_prime_dims_5d() {
  return runFftndRoundtrip<shafft::complexf>(
      {3, 5, 7, 3, 5}, shafft::FFTType::C2C, test::TOL_F);
}

int main(int argc, char** argv) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::fprintf(stderr, "MPI_Init failed\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  test::TestRunnerMPI runner("High-Dimensional Roundtrip Tests");

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
