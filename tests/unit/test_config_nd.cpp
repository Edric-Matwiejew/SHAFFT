/**
 * @file test_config_nd.cpp
 * @brief Unit tests for N-D config object lifecycle (C and C++ APIs).
 */
#include "test_utils.hpp"

#include <cstring>
#include <mpi.h>
#include <shafft/shafft.h>
#include <shafft/shafft.hpp>
#include <vector>

// ---- C API tests -----------------------------------------------------------

/// Init / release roundtrip.
static bool testConfigNDInitRelease() {
  shafft_nd_config_t cfg = {};
  size_t shape[] = {8, 8, 8};
  int rc = shafftConfigNDInit(&cfg,
                              3,
                              shape,
                              SHAFFT_C2C,
                              nullptr,
                              0,
                              SHAFFT_MAXIMIZE_NDA,
                              SHAFFT_LAYOUT_REDISTRIBUTED,
                              0,
                              MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  if (cfg.structSize != sizeof(shafft_nd_config_t))
    return false;
  if (cfg.ndim != 3)
    return false;
  if (cfg.globalShape == nullptr)
    return false;
  if (cfg.hintCommDims == nullptr)
    return false;
  if (cfg.commDims == nullptr)
    return false;
  if (cfg.initial.subsize == nullptr)
    return false;
  if (cfg.output.subsize == nullptr)
    return false;
  shafftConfigNDRelease(&cfg);
  if (cfg.globalShape != nullptr)
    return false; // must be nulled
  if (cfg.structSize != 0)
    return false; // must be zeroed
  return true;
}

/// Double init must fail (SHAFFT_ERR_INVALID_STATE).
static bool testConfigNDDoubleInit() {
  shafft_nd_config_t cfg = {};
  size_t shape[] = {4, 4};
  int rc1 = shafftConfigNDInit(&cfg,
                               2,
                               shape,
                               SHAFFT_C2C,
                               nullptr,
                               0,
                               SHAFFT_MAXIMIZE_NDA,
                               SHAFFT_LAYOUT_REDISTRIBUTED,
                               0,
                               MPI_COMM_WORLD);
  if (rc1 != 0)
    return false;
  int rc2 = shafftConfigNDInit(&cfg,
                               2,
                               shape,
                               SHAFFT_C2C,
                               nullptr,
                               0,
                               SHAFFT_MAXIMIZE_NDA,
                               SHAFFT_LAYOUT_REDISTRIBUTED,
                               0,
                               MPI_COMM_WORLD); // second init should fail
  shafftConfigNDRelease(&cfg);
  return rc2 != 0;
}

/// Null pointer rejected.
static bool testConfigNDNullPtr() {
  size_t shape[] = {8, 8, 8};
  int rc = shafftConfigNDInit(nullptr,
                              3,
                              shape,
                              SHAFFT_C2C,
                              nullptr,
                              0,
                              SHAFFT_MAXIMIZE_NDA,
                              SHAFFT_LAYOUT_REDISTRIBUTED,
                              0,
                              MPI_COMM_WORLD);
  return rc != 0;
}

/// Init+resolve a simple 3-D config (init now does resolve).
static bool testConfigNDResolve() {
  shafft_nd_config_t cfg = {};
  size_t shape[] = {8, 8, 8};
  int rc = shafftConfigNDInit(&cfg,
                              3,
                              shape,
                              SHAFFT_C2C,
                              nullptr,
                              0,
                              SHAFFT_MAXIMIZE_NDA,
                              SHAFFT_LAYOUT_REDISTRIBUTED,
                              0,
                              MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  // Check resolved flag is set (init does resolve)
  if (!(cfg.flags & SHAFFT_CONFIG_RESOLVED)) {
    shafftConfigNDRelease(&cfg);
    return false;
  }

  // Check nda is set
  if (cfg.nda < 0 || cfg.nda > 3) {
    shafftConfigNDRelease(&cfg);
    return false;
  }

  // Check commSize > 0
  if (cfg.commSize <= 0) {
    shafftConfigNDRelease(&cfg);
    return false;
  }

  // Check local elements > 0 for single rank
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  if (worldSize == 1 && cfg.initial.localElements != 512) {
    shafftConfigNDRelease(&cfg);
    return false;
  }

  // Topology: hostname must be set
  if (cfg.hostname == nullptr || cfg.hostnameLen == 0) {
    shafftConfigNDRelease(&cfg);
    return false;
  }

  shafftConfigNDRelease(&cfg);
  return true;
}

/// Size overflow rejected (dimension > INT_MAX).
static bool testConfigNDOverflow() {
  shafft_nd_config_t cfg = {};
  size_t shape[] = {static_cast<size_t>(std::numeric_limits<int>::max()) + 1};
  int rc = shafftConfigNDInit(&cfg,
                              1,
                              shape,
                              SHAFFT_C2C,
                              nullptr,
                              0,
                              SHAFFT_MAXIMIZE_NDA,
                              SHAFFT_LAYOUT_REDISTRIBUTED,
                              0,
                              MPI_COMM_WORLD);
  shafftConfigNDRelease(&cfg);
  return rc != 0; // must fail with overflow
}

// ---- C++ API tests ---------------------------------------------------------

/// ConfigND RAII wrapper basic lifecycle.
static bool testConfigNDCpp() {
  shafft::ConfigND cfg(
      {8, 8, 8}, shafft::FFTType::C2C, {}, 0, shafft::DecompositionStrategy::MAXIMIZE_NDA);
  if (cfg.status() != 0)
    return false;
  if (!cfg.isResolved())
    return false;

  const auto& c = cfg.cStruct();
  if (c.nda < 0)
    return false;
  if (c.commSize <= 0)
    return false;
  return true;
}

/// ConfigND move semantics.
static bool testConfigNDMove() {
  shafft::ConfigND a({16, 16}, shafft::FFTType::C2C);
  if (a.status() != 0)
    return false;

  shafft::ConfigND b(std::move(a));
  if (b.status() != 0)
    return false;
  // a should be in moved-from state
  if (a.status() == 0)
    return false; // NOLINT

  if (!b.isResolved())
    return false;
  return true;
}

/// FFTND init from config object.
static bool testFFTNDInitFromConfig() {
  shafft_nd_config_t cfg = {};
  size_t shape[] = {8, 8, 8};
  int rc = shafftConfigNDInit(&cfg,
                              3,
                              shape,
                              SHAFFT_C2C,
                              nullptr,
                              0,
                              SHAFFT_MAXIMIZE_NDA,
                              SHAFFT_LAYOUT_REDISTRIBUTED,
                              0,
                              MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  shafft::FFTND fft;
  rc = fft.init(cfg);
  if (rc != 0) {
    shafftConfigNDRelease(&cfg);
    return false;
  }

  if (!fft.isConfigured()) {
    shafftConfigNDRelease(&cfg);
    return false;
  }
  if (fft.allocSize() == 0) {
    shafftConfigNDRelease(&cfg);
    return false;
  }

  shafftConfigNDRelease(&cfg);
  return true;
}

int main(int argc, char* argv[]) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::fprintf(stderr, "MPI_Init failed\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  test::TestRunner runner("ConfigND Unit Tests");

  runner.run("C: init/release roundtrip", testConfigNDInitRelease);
  runner.run("C: double init rejected", testConfigNDDoubleInit);
  runner.run("C: null pointer rejected", testConfigNDNullPtr);
  runner.run("C: resolve 3D", testConfigNDResolve);
  runner.run("C: size overflow rejected", testConfigNDOverflow);
  runner.run("C++: ConfigND lifecycle", testConfigNDCpp);
  runner.run("C++: ConfigND move", testConfigNDMove);
  runner.run("C++: FFTND initFromConfig", testFFTNDInitFromConfig);

  int result = runner.finalize();
  MPI_Finalize();
  return result;
}
