/**
 * @file test_config_1d.cpp
 * @brief Unit tests for 1-D config object lifecycle (C and C++ APIs).
 */
#include "test_utils.hpp"

#include <cstring>
#include <limits>
#include <mpi.h>
#include <shafft/shafft.h>
#include <shafft/shafft.hpp>

// ---- C API tests -----------------------------------------------------------

/// Init / release roundtrip.
static bool testConfig1DInitRelease() {
  shafft_1d_config_t cfg = {};
  int rc = shafftConfig1DInit(&cfg, 1024, SHAFFT_C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  if (cfg.structSize != sizeof(shafft_1d_config_t))
    return false;
  if (cfg.globalSize != 1024)
    return false;
  if (cfg.precision != SHAFFT_C2C)
    return false;
  shafftConfig1DRelease(&cfg);
  if (cfg.structSize != 0)
    return false;
  return true;
}

/// Null pointer rejected.
static bool testConfig1DNullPtr() {
  return shafftConfig1DInit(nullptr, 1024, SHAFFT_C2C, MPI_COMM_WORLD) != 0;
}

/// Zero global size rejected.
static bool testConfig1DZeroSize() {
  shafft_1d_config_t cfg = {};
  return shafftConfig1DInit(&cfg, 0, SHAFFT_C2C, MPI_COMM_WORLD) != 0;
}

/// Double init rejected.
static bool testConfig1DDoubleInit() {
  shafft_1d_config_t cfg = {};
  shafftConfig1DInit(&cfg, 64, SHAFFT_C2C, MPI_COMM_WORLD);
  int rc2 = shafftConfig1DInit(&cfg, 64, SHAFFT_C2C, MPI_COMM_WORLD);
  shafftConfig1DRelease(&cfg);
  return rc2 != 0;
}

/// Init+resolve a 1-D config (init now does resolve).
static bool testConfig1DResolve() {
  shafft_1d_config_t cfg = {};
  int rc = shafftConfig1DInit(&cfg, 128, SHAFFT_C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  // Init does resolve — check resolved flag
  if (!(cfg.flags & SHAFFT_CONFIG_RESOLVED)) {
    shafftConfig1DRelease(&cfg);
    return false;
  }

  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  // For single rank: localSize should == globalSize
  if (worldSize == 1) {
    if (cfg.initial.localSize != 128) {
      shafftConfig1DRelease(&cfg);
      return false;
    }
    if (cfg.initial.localStart != 0) {
      shafftConfig1DRelease(&cfg);
      return false;
    }
  }

  if (cfg.allocElements == 0) {
    shafftConfig1DRelease(&cfg);
    return false;
  }
  if (cfg.hostname == nullptr) {
    shafftConfig1DRelease(&cfg);
    return false;
  }

  shafftConfig1DRelease(&cfg);
  return true;
}

/// Size overflow rejected.
static bool testConfig1DOverflow() {
  shafft_1d_config_t cfg = {};
  size_t big = static_cast<size_t>(std::numeric_limits<int>::max()) + 1;
  int rc = shafftConfig1DInit(&cfg, big, SHAFFT_C2C, MPI_COMM_WORLD);
  shafftConfig1DRelease(&cfg);
  return rc != 0;
}

// ---- C++ API tests ---------------------------------------------------------

/// Config1D RAII wrapper basic lifecycle.
static bool testConfig1DCpp() {
  shafft::Config1D cfg(128, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (cfg.status() != 0)
    return false;
  if (!cfg.isResolved())
    return false;

  const auto& c = cfg.cStruct();
  if (c.allocElements == 0)
    return false;
  return true;
}

/// Config1D move semantics.
static bool testConfig1DMove() {
  shafft::Config1D a(64, shafft::FFTType::Z2Z, MPI_COMM_WORLD);
  if (a.status() != 0)
    return false;

  shafft::Config1D b(std::move(a));
  if (b.status() != 0)
    return false;
  if (a.status() == 0)
    return false; // NOLINT(moved-from)

  if (!b.isResolved())
    return false;
  return true;
}

/// FFT1D init from config object.
static bool testFFT1DInitFromConfig() {
  shafft_1d_config_t cfg = {};
  int rc = shafftConfig1DInit(&cfg, 64, SHAFFT_C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  shafft::FFT1D fft;
  rc = fft.init(cfg);
  if (rc != 0) {
    shafftConfig1DRelease(&cfg);
    return false;
  }

  if (!fft.isConfigured()) {
    shafftConfig1DRelease(&cfg);
    return false;
  }
  if (fft.allocSize() == 0) {
    shafftConfig1DRelease(&cfg);
    return false;
  }

  shafftConfig1DRelease(&cfg);
  return true;
}

int main(int argc, char* argv[]) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::fprintf(stderr, "MPI_Init failed\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  test::TestRunner runner("Config1D Unit Tests");

  runner.run("C: init/release roundtrip", testConfig1DInitRelease);
  runner.run("C: null pointer rejected", testConfig1DNullPtr);
  runner.run("C: zero size rejected", testConfig1DZeroSize);
  runner.run("C: double init rejected", testConfig1DDoubleInit);
  runner.run("C: resolve", testConfig1DResolve);
  runner.run("C: size overflow rejected", testConfig1DOverflow);
  runner.run("C++: Config1D lifecycle", testConfig1DCpp);
  runner.run("C++: Config1D move", testConfig1DMove);
  runner.run("C++: FFT1D initFromConfig", testFFT1DInitFromConfig);

  int result = runner.finalize();
  MPI_Finalize();
  return result;
}
