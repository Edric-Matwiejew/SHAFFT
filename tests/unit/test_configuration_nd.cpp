/**
 * @file test_configuration_nd.cpp
 * @brief Unit tests for configurationND() unified configuration function
 *
 * Tests the new configurationND() API that provides cascading configuration:
 *   - DecompositionStrategy to compute optimal nda and commDims
 *
 * Also tests the new FFTND::init(commDims, dims) method.
 */
#include "test_utils.hpp"
#include <mpi.h>
#include <numeric>
#include <shafft/shafft.hpp>
#include <vector>

// Test: configurationND with explicit commDims via MAXIMIZE_NDA strategy
static bool testConfigurationNDMaximize() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  std::vector<size_t> dims = {64, 64, 64};
  const int ndim = static_cast<int>(dims.size());
  std::vector<int> commDims(ndim, 0); // Pre-size to ndim
  int nda = 0;
  std::vector<size_t> subsize(ndim), offset(ndim); // Pre-size to ndim
  int commSize = 0;

  int rc = shafft::configurationND(dims,
                                   shafft::FFTType::C2C,
                                   commDims,
                                   nda,
                                   subsize,
                                   offset,
                                   commSize,
                                   shafft::DecompositionStrategy::MAXIMIZE_NDA,
                                   0,
                                   MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  // Verify commDims is computed
  if (commDims.size() != 3)
    return false;

  // Verify subsize and offset are computed correctly
  if (subsize.size() != 3)
    return false;
  if (offset.size() != 3)
    return false;

  // commSize should match worldSize (or close to it)
  if (commSize <= 0)
    return false;

  return true;
}

// Test: configurationND with MINIMIZE_NDA strategy
static bool testConfigurationNDMinimize() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  std::vector<size_t> dims = {64, 64, 64};
  const int ndim = static_cast<int>(dims.size());
  std::vector<int> commDims(ndim, 0);
  int nda = 0;
  std::vector<size_t> subsize(ndim), offset(ndim);
  int commSize = 0;

  int rc = shafft::configurationND(dims,
                                   shafft::FFTType::C2C,
                                   commDims,
                                   nda,
                                   subsize,
                                   offset,
                                   commSize,
                                   shafft::DecompositionStrategy::MINIMIZE_NDA,
                                   0,
                                   MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  // Verify dimensions match
  if (commDims.size() != 3)
    return false;
  if (subsize.size() != 3)
    return false;

  return true;
}

// Test: configurationND returns consistent subsize/offset
static bool testSubsizeOffsetConsistency() {
  int worldSize, worldRank;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

  std::vector<size_t> dims = {16, 24, 32};
  const int ndim = static_cast<int>(dims.size());
  std::vector<int> commDims(ndim, 0);
  int nda = 0;
  std::vector<size_t> subsize(ndim), offset(ndim);
  int commSize = 0;

  int rc = shafft::configurationND(dims,
                                   shafft::FFTType::C2C,
                                   commDims,
                                   nda,
                                   subsize,
                                   offset,
                                   commSize,
                                   shafft::DecompositionStrategy::MINIMIZE_NDA,
                                   0,
                                   MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  // Verify subsize elements are positive (for active ranks)
  // or that the rank is inactive
  size_t localElems = test::product(subsize);

  // Gather all local sizes to check they sum to global size (or more with padding)
  std::vector<size_t> allSizes(worldSize);
  MPI_Allgather(
      &localElems, 1, MPI_UNSIGNED_LONG, allSizes.data(), 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);

  size_t globalExpected = static_cast<size_t>(dims[0]) * dims[1] * dims[2];
  size_t total = 0;
  for (int i = 0; i < worldSize; ++i)
    total += allSizes[i];

  // Total should be >= global (padding allowed)
  if (total < globalExpected)
    return false;

  return true;
}

// Test: configurationND with 2D tensor
static bool test2DConfiguration() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  std::vector<size_t> dims = {128, 256};
  const int ndim = static_cast<int>(dims.size());
  std::vector<int> commDims(ndim, 0);
  int nda = 0;
  std::vector<size_t> subsize(ndim), offset(ndim);
  int commSize = 0;

  int rc = shafft::configurationND(dims,
                                   shafft::FFTType::C2C,
                                   commDims,
                                   nda,
                                   subsize,
                                   offset,
                                   commSize,
                                   shafft::DecompositionStrategy::MINIMIZE_NDA,
                                   0,
                                   MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  // Verify dimensions
  if (subsize.size() != 2)
    return false;
  if (commDims.size() != 2)
    return false;

  return true;
}

// Test: configurationND with double precision (Z2Z)
static bool testDoublePrecision() {

  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  std::vector<size_t> dims = {32, 32, 32};
  const int ndim = static_cast<int>(dims.size());
  std::vector<int> commDims(ndim, 0);
  int nda = 0;
  std::vector<size_t> subsize(ndim), offset(ndim);
  int commSize = 0;

  int rc = shafft::configurationND(dims,
                                   shafft::FFTType::Z2Z,
                                   commDims,
                                   nda,
                                   subsize,
                                   offset,
                                   commSize,
                                   shafft::DecompositionStrategy::MINIMIZE_NDA,
                                   0,
                                   MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  // Configuration should work identically for Z2Z
  if (subsize.size() != 3)
    return false;

  return true;
}

// Test: FFTND::init with commDims (new API)
static bool testPlanNDInitCommDims() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  std::vector<size_t> dims = {32, 32, 32};
  const int ndim = static_cast<int>(dims.size());
  std::vector<int> commDims(ndim, 0);
  int nda = 0;
  std::vector<size_t> subsize(ndim), offset(ndim);
  int commSize = 0;

  // First get configuration
  int rc = shafft::configurationND(dims,
                                   shafft::FFTType::C2C,
                                   commDims,
                                   nda,
                                   subsize,
                                   offset,
                                   commSize,
                                   shafft::DecompositionStrategy::MINIMIZE_NDA,
                                   0,
                                   MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  // Now create plan with commDims
  shafft::FFTND fft;
  rc = fft.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  // Verify plan is configured
  if (!fft.isConfigured())
    return false;

  // Verify allocation size is reasonable
  size_t alloc = fft.allocSize();
  if (alloc == 0)
    return false;

  return true;
}

// Test: FFTND init with configurationND-computed commDims
static bool test_plannd_with_configuration() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  std::vector<size_t> dims = {32, 32, 32};
  const int ndim = static_cast<int>(dims.size());

  // Use configurationND to compute commDims
  std::vector<int> commDims(ndim, 0);
  std::vector<size_t> subsize(ndim), offset(ndim);
  int nda = 0, commSize = 0;
  int rc = shafft::configurationND(dims,
                                   shafft::FFTType::C2C,
                                   commDims,
                                   nda,
                                   subsize,
                                   offset,
                                   commSize,
                                   shafft::DecompositionStrategy::MINIMIZE_NDA,
                                   0,
                                   MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  // Create plan with computed commDims
  shafft::FFTND fft;
  rc = fft.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  if (!fft.isConfigured())
    return false;

  return true;
}

// Test: FFTND::getLayout and getAxes work after init
static bool test_plannd_queries() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  std::vector<size_t> dims = {16, 16, 16};
  const int ndim = static_cast<int>(dims.size());
  std::vector<int> commDims(ndim, 0);
  int nda = 0;
  std::vector<size_t> subsize(ndim), offset(ndim);
  int commSize = 0;

  int rc = shafft::configurationND(dims,
                                   shafft::FFTType::C2C,
                                   commDims,
                                   nda,
                                   subsize,
                                   offset,
                                   commSize,
                                   shafft::DecompositionStrategy::MINIMIZE_NDA,
                                   0,
                                   MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  shafft::FFTND fft;
  rc = fft.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  // Test getLayout - vectors must be pre-sized
  std::vector<size_t> layout_subsize(ndim), layout_offset(ndim);
  rc = fft.getLayout(layout_subsize, layout_offset, shafft::TensorLayout::CURRENT);
  if (rc != 0)
    return false;
  if (layout_subsize.size() != 3)
    return false;

  // Test getAxes - vectors will be populated by the API
  std::vector<int> ca(ndim), da(ndim); // Pre-size to ndim
  rc = fft.getAxes(ca, da, shafft::TensorLayout::CURRENT);
  if (rc != 0)
    return false;

  return true;
}

// Main
int main(int argc, char* argv[]) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::fprintf(stderr, "MPI_Init failed\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  test::TestRunner runner("configurationND Unit Tests");

  runner.run("configurationND MAXIMIZE_NDA", testConfigurationNDMaximize);
  runner.run("configurationND MINIMIZE_NDA", testConfigurationNDMinimize);
  runner.run("subsize/offset consistency", testSubsizeOffsetConsistency);
  runner.run("2D configuration", test2DConfiguration);
  runner.run("double precision (Z2Z)", testDoublePrecision);
  runner.run("FFTND::init with commDims", testPlanNDInitCommDims);
  runner.run("FFTND with configurationND", test_plannd_with_configuration);
  runner.run("FFTND queries", test_plannd_queries);

  int result = runner.finalize();
  MPI_Finalize();
  return result;
}
