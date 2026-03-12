/**
 * @file test_library_info.cpp
 * @brief Unit tests for library information functions (C++ API)
 *
 * Tests getVersion(), getVersionString(), getBackendName() which are
 * compile-time constants. True unit tests - no MPI communication.
 */
#include "test_utils.hpp"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <shafft/shafft.hpp>
#include <sstream>
#include <string>

// Test: C++ Version struct has valid components
static bool test_cpp_version_valid() {
  shafft::Version v = shafft::getVersion();

  // Version components should be non-negative
  if (v.major < 0) {
    std::cerr << "Invalid major version: " << v.major << "\n";
    return false;
  }
  if (v.minor < 0) {
    std::cerr << "Invalid minor version: " << v.minor << "\n";
    return false;
  }
  if (v.patch < 0) {
    std::cerr << "Invalid patch version: " << v.patch << "\n";
    return false;
  }

  // At least one component should be > 0 (we're not at 0.0.0)
  // Actually 0.0.1 is valid, so just check not all are unreasonably large
  if (v.major > 100 || v.minor > 100 || v.patch > 1000) {
    std::cerr << "Suspiciously large version: " << v.major << "." << v.minor << "." << v.patch
              << "\n";
    return false;
  }

  return true;
}

// Test: Version string matches version components
static bool test_version_string_matches_components() {
  shafft::Version v = shafft::getVersion();
  const char* vstr = shafft::getVersionString();

  if (!vstr) {
    std::cerr << "getVersionString() returned null\n";
    return false;
  }

  // Build expected numeric prefix
  char expected[64];
  snprintf(expected, sizeof(expected), "%d.%d.%d", v.major, v.minor, v.patch);

  const std::string prefix(expected);
  const std::string actual(vstr);
  const std::string suffix = SHAFFT_VERSION_SUFFIX; // may be empty or start with '-'

  // Accept exact match or numeric prefix + configured suffix
  if (actual == prefix)
    return true;
  if (!suffix.empty() && actual == prefix + suffix)
    return true;

  std::cerr << "Version mismatch: getVersionString()='" << actual << "' but components give '"
            << prefix << "' (suffix='" << suffix << "')\n";
  return false;
}

// Test: Backend name is valid
static bool test_backend_name_valid() {
  const char* name = shafft::getBackendName();

  if (!name) {
    std::cerr << "getBackendName() returned null\n";
    return false;
  }

  // Should be one of the known backends
  bool valid = (std::strcmp(name, "FFTW") == 0) || (std::strcmp(name, "hipFFT") == 0);

  if (!valid) {
    std::cerr << "Unknown backend name: '" << name << "'\n";
    std::cerr << "Expected 'FFTW' or 'hipFFT'\n";
    return false;
  }

  return true;
}

// Test: Version string is stable across calls
static bool test_version_string_stable() {
  const char* v1 = shafft::getVersionString();
  const char* v2 = shafft::getVersionString();
  const char* v3 = shafft::getVersionString();

  // Should return same pointer (static storage)
  if (v1 != v2 || v2 != v3) {
    std::cerr << "Version string pointer not stable\n";
    return false;
  }

  return true;
}

// Test: Backend name is stable across calls
static bool test_backend_name_stable() {
  const char* b1 = shafft::getBackendName();
  const char* b2 = shafft::getBackendName();
  const char* b3 = shafft::getBackendName();

  // Should return same pointer (static/literal storage)
  if (b1 != b2 || b2 != b3) {
    std::cerr << "Backend name pointer not stable\n";
    return false;
  }

  return true;
}

// Test: Version string format is valid (X.Y.Z)
static bool test_version_string_format() {
  const char* vstr = shafft::getVersionString();

  int major, minor, patch;
  int parsed = sscanf(vstr, "%d.%d.%d", &major, &minor, &patch);

  if (parsed != 3) {
    std::cerr << "Version string '" << vstr << "' does not match X.Y.Z format\n";
    return false;
  }

  // Allow optional suffix beginning with '-' after the numeric portion
  char rebuilt[64];
  snprintf(rebuilt, sizeof(rebuilt), "%d.%d.%d", major, minor, patch);

  const std::string actual(vstr);
  const std::string base(rebuilt);

  if (actual == base)
    return true;
  if (actual.rfind(base + "-", 0) == 0)
    return true; // base followed by '-' suffix

  std::cerr << "Version string has unexpected content: '" << actual << "' (expected '" << base
            << "' or base+'-suffix')\n";
  return false;
}

// Main
int main(int argc, char** argv) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::fprintf(stderr, "MPI_Init failed\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int ret = 0;
  if (rank == 0) {
    test::TestRunner runner("Library Info Unit Tests (C++ API)");
    runner.run("cpp_version_valid", test_cpp_version_valid);
    runner.run("version_string_matches_components", test_version_string_matches_components);
    runner.run("backend_name_valid", test_backend_name_valid);
    runner.run("version_string_stable", test_version_string_stable);
    runner.run("backend_name_stable", test_backend_name_stable);
    runner.run("version_string_format", test_version_string_format);
    ret = runner.finalize();
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return ret;
}
