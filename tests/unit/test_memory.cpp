/**
 * @file test_memory.cpp
 * @brief Unit tests for portable memory allocation and copy helpers (C++ API)
 *
 * Tests allocBuffer(), freeBuffer(), copyToBuffer(), copyFromBuffer().
 * These work identically on CPU (FFTW) and GPU (hipFFT) backends.
 */
#include "test_utils.hpp"
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <shafft/shafft.hpp>
#include <vector>

// Test: Single-precision allocation returns non-null
static bool test_alloc_float_nonnull() {
  shafft::complexf* buf = nullptr;
  int rc = shafft::allocBuffer(1024, &buf);

  if (rc != 0) {
    std::cerr << "allocBuffer(1024) failed with rc=" << rc << "\n";
    return false;
  }

  if (!buf) {
    std::cerr << "allocBuffer returned success but buf is null\n";
    return false;
  }

  // Clean up
  (void)shafft::freeBuffer(buf);
  return true;
}

// Test: Double-precision allocation returns non-null
static bool test_alloc_double_nonnull() {
  shafft::complexd* buf = nullptr;
  int rc = shafft::allocBuffer(1024, &buf);

  if (rc != 0) {
    std::cerr << "allocBuffer(1024) for double failed with rc=" << rc << "\n";
    return false;
  }

  if (!buf) {
    std::cerr << "allocBuffer returned success but buf is null\n";
    return false;
  }

  (void)shafft::freeBuffer(buf);
  return true;
}

// Test: Zero-size allocation behavior
// Expected: Either succeeds with nullptr/freeable pointer, or returns error code.
// Must not crash either way.
static bool test_alloc_zero_size() {
  shafft::complexf* buf = nullptr;
  int rc = shafft::allocBuffer(0, &buf);

  if (rc == 0) {
    // Success case: buf may be nullptr or non-null (implementation-defined)
    // If non-null, must be freeable without error
    if (buf != nullptr) {
      int freeRc = shafft::freeBuffer(buf);
      if (freeRc != 0) {
        std::cerr << "freeBuffer failed on zero-size allocation: rc=" << freeRc << "\n";
        return false;
      }
    }
    return true;
  } else {
    // Error case: Also acceptable - some backends reject zero-size
    // Just verify buf wasn't modified to garbage
    if (buf != nullptr) {
      std::cerr << "allocBuffer(0) failed but set buf to non-null\n";
      return false;
    }
    return true;
  }
}

// Test: Free handles null gracefully
// Expected: freeBuffer(nullptr) should return success (rc==0) per common convention.
static bool test_free_null_safe() {
  shafft::complexf* nullF = nullptr;
  shafft::complexd* nullD = nullptr;

  int rc1 = shafft::freeBuffer(nullF);
  int rc2 = shafft::freeBuffer(nullD);

  // Both should succeed - freeing nullptr is a no-op
  if (rc1 != 0) {
    std::cerr << "freeBuffer(nullptr) for float failed: rc=" << rc1 << "\n";
    return false;
  }
  if (rc2 != 0) {
    std::cerr << "freeBuffer(nullptr) for double failed: rc=" << rc2 << "\n";
    return false;
  }

  return true;
}

// Test: Roundtrip copy preserves data (single precision)
static bool test_copy_roundtrip_float() {
  const size_t N = 256;

  // Host data with known pattern
  std::vector<shafft::complexf> host_src(N);
  for (size_t i = 0; i < N; ++i) {
    host_src[i] = {static_cast<float>(i), static_cast<float>(N - i)};
  }

  // Allocate device/backend buffer
  shafft::complexf* buf = nullptr;
  int rc = shafft::allocBuffer(N, &buf);
  if (rc != 0 || !buf) {
    std::cerr << "Failed to allocate buffer\n";
    return false;
  }

  // Copy host -> buffer
  rc = shafft::copyToBuffer(buf, host_src.data(), N);
  if (rc != 0) {
    std::cerr << "copyToBuffer failed with rc=" << rc << "\n";
    (void)shafft::freeBuffer(buf);
    return false;
  }

  // Copy buffer -> host (different destination)
  std::vector<shafft::complexf> host_dst(N, {-1.0f, -1.0f});
  rc = shafft::copyFromBuffer(host_dst.data(), buf, N);
  if (rc != 0) {
    std::cerr << "copyFromBuffer failed with rc=" << rc << "\n";
    (void)shafft::freeBuffer(buf);
    return false;
  }

  // Verify data matches
  for (size_t i = 0; i < N; ++i) {
    if (host_dst[i].real() != host_src[i].real() || host_dst[i].imag() != host_src[i].imag()) {
      std::cerr << "Mismatch at index " << i << ": " << "src=(" << host_src[i].real() << ","
                << host_src[i].imag() << ") " << "dst=(" << host_dst[i].real() << ","
                << host_dst[i].imag() << ")\n";
      (void)shafft::freeBuffer(buf);
      return false;
    }
  }

  (void)shafft::freeBuffer(buf);
  return true;
}

// Test: Roundtrip copy preserves data (double precision)
static bool test_copy_roundtrip_double() {
  const size_t N = 256;

  // Host data with known pattern (use values that need double precision)
  std::vector<shafft::complexd> host_src(N);
  for (size_t i = 0; i < N; ++i) {
    // Use values that would lose precision in float
    host_src[i] = {1.0 + 1e-10 * static_cast<double>(i), 2.0 + 1e-10 * static_cast<double>(N - i)};
  }

  shafft::complexd* buf = nullptr;
  int rc = shafft::allocBuffer(N, &buf);
  if (rc != 0 || !buf) {
    std::cerr << "Failed to allocate double buffer\n";
    return false;
  }

  rc = shafft::copyToBuffer(buf, host_src.data(), N);
  if (rc != 0) {
    (void)shafft::freeBuffer(buf);
    return false;
  }

  std::vector<shafft::complexd> host_dst(N, {-1.0, -1.0});
  rc = shafft::copyFromBuffer(host_dst.data(), buf, N);
  if (rc != 0) {
    (void)shafft::freeBuffer(buf);
    return false;
  }

  // Verify exact match (no floating point operations, just copy)
  for (size_t i = 0; i < N; ++i) {
    if (host_dst[i].real() != host_src[i].real() || host_dst[i].imag() != host_src[i].imag()) {
      std::cerr << "Double mismatch at index " << i << "\n";
      (void)shafft::freeBuffer(buf);
      return false;
    }
  }

  (void)shafft::freeBuffer(buf);
  return true;
}

// Test: Large allocation (stress test)
static bool test_large_allocation() {
  // 64 MB worth of complex floats
  const size_t N = 8 * 1024 * 1024; // 8M elements = 64MB

  shafft::complexf* buf = nullptr;
  int rc = shafft::allocBuffer(N, &buf);

  if (rc != 0) {
    // Large allocation may fail on memory-constrained systems
    // That's acceptable - just shouldn't crash
    std::cout << "(skipped - allocation failed, may be memory-constrained)\n";
    return true;
  }

  if (!buf) {
    std::cerr << "Large allocation returned success but null pointer\n";
    return false;
  }

  (void)shafft::freeBuffer(buf);
  return true;
}

// Test: Multiple allocations and frees
static bool test_multiple_alloc_free() {
  const int NUM_BUFFERS = 10;
  std::vector<shafft::complexf*> buffers(NUM_BUFFERS, nullptr);

  // Allocate all
  for (int i = 0; i < NUM_BUFFERS; ++i) {
    int rc = shafft::allocBuffer(1024 * (i + 1), &buffers[i]);
    if (rc != 0 || !buffers[i]) {
      std::cerr << "Failed to allocate buffer " << i << "\n";
      // Clean up already allocated
      for (int j = 0; j < i; ++j) {
        (void)shafft::freeBuffer(buffers[j]);
      }
      return false;
    }
  }

  // Free in reverse order
  for (int i = NUM_BUFFERS - 1; i >= 0; --i) {
    int rc = shafft::freeBuffer(buffers[i]);
    if (rc != 0) {
      std::cerr << "Failed to free buffer " << i << "\n";
      return false;
    }
  }

  return true;
}

// Test: Partial copy (subset of buffer)
static bool test_partial_copy() {
  const size_t N = 1024;
  const size_t SUBSET = 100;

  std::vector<shafft::complexf> host_src(N);
  for (size_t i = 0; i < N; ++i) {
    host_src[i] = {static_cast<float>(i), 0.0f};
  }

  shafft::complexf* buf = nullptr;
  int rc = shafft::allocBuffer(N, &buf);
  if (rc != 0 || !buf) {
    return false;
  }

  // Copy only first SUBSET elements
  rc = shafft::copyToBuffer(buf, host_src.data(), SUBSET);
  if (rc != 0) {
    (void)shafft::freeBuffer(buf);
    return false;
  }

  std::vector<shafft::complexf> host_dst(SUBSET, {-1.0f, -1.0f});
  rc = shafft::copyFromBuffer(host_dst.data(), buf, SUBSET);
  if (rc != 0) {
    (void)shafft::freeBuffer(buf);
    return false;
  }

  // Verify subset matches
  for (size_t i = 0; i < SUBSET; ++i) {
    if (host_dst[i].real() != host_src[i].real()) {
      std::cerr << "Partial copy mismatch at " << i << "\n";
      (void)shafft::freeBuffer(buf);
      return false;
    }
  }

  (void)shafft::freeBuffer(buf);
  return true;
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
    test::TestRunner runner("Memory Helpers Unit Tests (C++ API)");
    runner.run("alloc_float_nonnull", test_alloc_float_nonnull);
    runner.run("alloc_double_nonnull", test_alloc_double_nonnull);
    runner.run("alloc_zero_size", test_alloc_zero_size);
    runner.run("free_null_safe", test_free_null_safe);
    runner.run("copy_roundtrip_float", test_copy_roundtrip_float);
    runner.run("copy_roundtrip_double", test_copy_roundtrip_double);
    runner.run("large_allocation", test_large_allocation);
    runner.run("multiple_alloc_free", test_multiple_alloc_free);
    runner.run("partial_copy", test_partial_copy);
    ret = runner.finalize();
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return ret;
}
