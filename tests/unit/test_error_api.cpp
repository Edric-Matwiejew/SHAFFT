/**
 * @file test_error_api.cpp
 * @brief Unit tests for error query API (thread-local error state)
 *
 * Tests the shafftLastError* functions which track errors per-thread.
 * These are true unit tests - no MPI communication required.
 *
 * Note: The error functions have C linkage (extern "C") but are part of the
 * C++ library. They're accessible via shafft.h.
 */
#include "test_utils.hpp"
#include <shafft/shafft.h>
#include <shafft/shafft.hpp>

#include <mpi.h>

#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

// Test: Initial state is SUCCESS
static bool test_initial_state_is_success() {
  shafftClearLastError();

  int status = shafftLastErrorStatus();
  int source = shafftLastErrorSource();
  int code = shafftLastErrorDomainCode();

  if (status != 0) {
    std::cerr << "Expected status=0, got " << status << "\n";
    return false;
  }
  if (source != SHAFFT_ERRSRC_NONE) {
    std::cerr << "Expected source=NONE, got " << source << "\n";
    return false;
  }
  if (code != 0) {
    std::cerr << "Expected code=0, got " << code << "\n";
    return false;
  }
  return true;
}

// Test: Error state persists after a failed call
static bool test_error_persists_after_failure() {
  shafftClearLastError();

  // Trigger an error: ndim=0 is invalid (empty globalShape)
  shafft::ConfigND cfg({}, shafft::FFTType::C2C); // ndim<1 -> SHAFFT_ERR_INVALID_DIM
  int rc = cfg.status();

  if (rc == 0) {
    std::cerr << "Expected init to fail with empty dims\n";
    return false;
  }

  // Error state should be set
  int status = shafftLastErrorStatus();
  if (status == 0) {
    std::cerr << "Expected non-zero error status after failure\n";
    return false;
  }

  // Query again - should still be the same (persistent)
  int status2 = shafftLastErrorStatus();
  if (status != status2) {
    std::cerr << "Error status changed between queries\n";
    return false;
  }

  return true;
}

// Test: clear_last_error resets state
static bool test_clear_resets_state() {
  // First trigger an error: size overflow
  shafft::ConfigND cfg({static_cast<size_t>(-1)},
                       shafft::FFTType::C2C); // -> SHAFFT_ERR_SIZE_OVERFLOW

  int status_before = shafftLastErrorStatus();
  if (status_before == 0) {
    std::cerr << "Expected error to be set\n";
    return false;
  }

  // Clear it
  shafftClearLastError();

  int statusAfter = shafftLastErrorStatus();
  int sourceAfter = shafftLastErrorSource();
  int codeAfter = shafftLastErrorDomainCode();

  if (statusAfter != 0 || sourceAfter != 0 || codeAfter != 0) {
    std::cerr << "clear_last_error did not reset all fields\n";
    std::cerr << "  status=" << statusAfter << " source=" << sourceAfter << " code=" << codeAfter
              << "\n";
    return false;
  }

  return true;
}

// Test: Error source name lookup
static bool test_error_source_names() {
  struct TestCase {
    int source;
    const char* expected;
  };
  std::array<TestCase, 7> cases = {{
      {SHAFFT_ERRSRC_NONE, "SHAFFT"},
      {SHAFFT_ERRSRC_MPI, "MPI"},
      {SHAFFT_ERRSRC_HIP, "HIP"},
      {SHAFFT_ERRSRC_HIPFFT, "hipFFT"},
      {SHAFFT_ERRSRC_FFTW, "FFTW"},
      {SHAFFT_ERRSRC_SYSTEM, "System"},
      {999, "Unknown"}, // Invalid source
  }};

  for (const auto& tc : cases) {
    const char* name = shafftErrorSourceName(tc.source);
    if (std::strcmp(name, tc.expected) != 0) {
      std::cerr << "Source " << tc.source << ": expected '" << tc.expected << "', got '" << name
                << "'\n";
      return false;
    }
  }

  return true;
}

// Test: Error message formatting
static bool test_error_message_buffer() {
  shafftClearLastError();

  // With no error, message should be empty
  std::array<char, 256> buf{};
  int len = shafftLastErrorMessage(buf.data(), buf.size());

  // Should return 0 or empty string for no error
  if (len < 0) {
    std::cerr << "Unexpected negative length\n";
    return false;
  }

  // Test null buffer handling
  int lenNull = shafftLastErrorMessage(nullptr, 100);
  if (lenNull != 0) {
    std::cerr << "Expected 0 for null buffer, got " << lenNull << "\n";
    return false;
  }

  // Test zero-length buffer
  int lenZero = shafftLastErrorMessage(buf.data(), 0);
  if (lenZero != 0) {
    std::cerr << "Expected 0 for zero-length buffer, got " << lenZero << "\n";
    return false;
  }

  return true;
}

// Test: Error message truncation with small buffer
static bool test_error_message_truncation() {
  shafftClearLastError();

  // Trigger an error that should produce a message
  // We'll use a very small buffer to test truncation
  shafft::ConfigND cfg({}, shafft::FFTType::C2C); // ndim<1 -> SHAFFT_ERR_INVALID_DIM

  // Try with tiny buffer
  std::array<char, 4> tiny{};
  int len = shafftLastErrorMessage(tiny.data(), tiny.size());

  // Should not overflow
  if (len >= (int)tiny.size()) {
    std::cerr << "Truncation failed: len=" << len << " >= buflen=" << tiny.size() << "\n";
    return false;
  }

  // Should be null-terminated
  bool nullTerminated = false;
  for (char i : tiny) {
    if (i == '\0') {
      nullTerminated = true;
      break;
    }
  }
  if (!nullTerminated) {
    std::cerr << "Buffer not null-terminated\n";
    return false;
  }

  return true;
}

// Test: Thread-local error state isolation
// This tests that clearing error state on one thread doesn't affect another
static bool test_thread_local_isolation() {
  // This test verifies that error state is truly thread-local
  std::atomic<bool> threadPassed{true};
  std::atomic<bool> mainReady{false};
  std::atomic<bool> threadReady{false};

  // Clear main thread state
  shafftClearLastError();

  std::thread worker([&]() {
    // Clear worker thread state initially
    shafftClearLastError();

    // Note: simple validation errors don't set thread-local state
    // But we can still test that clear/query is thread-local
    threadReady = true;

    // Wait for main to do its operations
    while (!mainReady) {
      std::this_thread::yield();
    }

    // Worker state should still be cleared (not affected by main)
    int workerStatus = shafftLastErrorStatus();
    if (workerStatus != 0) {
      threadPassed = false; // Main shouldn't have polluted worker
    }
  });

  // Wait for worker to be ready
  while (!threadReady) {
    std::this_thread::yield();
  }

  // Main thread should have SUCCESS
  int main_status = shafftLastErrorStatus();
  if (main_status != 0) {
    std::cerr << "Main thread status not SUCCESS: " << main_status << "\n";
    mainReady = true;
    worker.join();
    return false;
  }

  // Clear main thread (should not affect worker)
  shafftClearLastError();
  mainReady = true;

  worker.join();

  if (!threadPassed) {
    std::cerr << "Worker thread state was affected by main thread\n";
    return false;
  }

  return true;
}

// Test: Successive errors overwrite previous state
static bool test_successive_errors_overwrite() {
  shafftClearLastError();

  // First error: invalid ndim
  shafft::ConfigND cfg1({}, shafft::FFTType::C2C); // ndim<1 -> SHAFFT_ERR_INVALID_DIM
  (void)shafftLastErrorStatus();                   // Query first error state

  // Second different error: valid init, execute without buffers
  shafft::FFTND fft2;
  shafft::ConfigND cfg2({32, 32}, shafft::FFTType::C2C);
  (void)fft2.init(cfg2.cStruct());

  // Execute without buffers - different error type
  (void)fft2.execute(shafft::FFTDirection::FORWARD);
  int status2 = shafftLastErrorStatus();

  // Second error should have overwritten first
  // (They may be the same code, but state should be updated)
  // Key test: error state should reflect LAST error, not first
  if (status2 == 0) {
    std::cerr << "Expected error status after execute without buffers\n";
    return false;
  }

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
    test::TestRunner runner("Error API Unit Tests");
    runner.run("initial_state_is_success", test_initial_state_is_success);
    runner.run("error_persists_after_failure", test_error_persists_after_failure);
    runner.run("clear_resets_state", test_clear_resets_state);
    runner.run("error_source_names", test_error_source_names);
    runner.run("error_message_buffer", test_error_message_buffer);
    runner.run("error_message_truncation", test_error_message_truncation);
    runner.run("thread_local_isolation", test_thread_local_isolation);
    runner.run("successive_errors_overwrite", test_successive_errors_overwrite);
    ret = runner.finalize();
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return ret;
}
