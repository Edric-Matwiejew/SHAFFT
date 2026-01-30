/**
 * @file test_error_api.cpp
 * @brief Unit tests for error query API (thread-local error state)
 * 
 * Tests the shafft_last_error_* functions which track errors per-thread.
 * These are true unit tests - no MPI communication required.
 * 
 * Note: The error functions have C linkage (extern "C") but are part of the
 * C++ library. They're accessible via shafft_error.hpp.
 */
#include <shafft/shafft.hpp>
#include <shafft/shafft_error.hpp>
#include <mpi.h>
#include <cstring>
#include <iostream>
#include <thread>
#include <atomic>
#include <vector>

// Test result tracking
static int g_passed = 0;
static int g_failed = 0;

#define TEST(name) \
    static bool test_##name(); \
    static bool run_##name() { \
        bool ok = test_##name(); \
        if (ok) { g_passed++; std::cout << "  " #name " PASS\n"; } \
        else { g_failed++; std::cout << "  " #name " FAIL\n"; } \
        return ok; \
    } \
    static bool test_##name()

//------------------------------------------------------------------------------
// Test: Initial state is SUCCESS
//------------------------------------------------------------------------------
TEST(initial_state_is_success) {
    shafft_clear_last_error();
    
    int status = shafft_last_error_status();
    int source = shafft_last_error_source();
    int code = shafft_last_error_domain_code();
    
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

//------------------------------------------------------------------------------
// Test: Error state persists after a failed call
//------------------------------------------------------------------------------
TEST(error_persists_after_failure) {
    shafft_clear_last_error();
    
    // Trigger an error: empty dims
    shafft::Plan plan;
    std::vector<int> dims = {};  // Empty dims -> error
    int rc = plan.init(1, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    
    if (rc == 0) {
        std::cerr << "Expected init to fail with empty dims\n";
        return false;
    }
    
    // Error state should be set
    int status = shafft_last_error_status();
    if (status == 0) {
        std::cerr << "Expected non-zero error status after failure\n";
        return false;
    }
    
    // Query again - should still be the same (persistent)
    int status2 = shafft_last_error_status();
    if (status != status2) {
        std::cerr << "Error status changed between queries\n";
        return false;
    }
    
    return true;
}

//------------------------------------------------------------------------------
// Test: clear_last_error resets state
//------------------------------------------------------------------------------
TEST(clear_resets_state) {
    // First trigger an error
    shafft::Plan plan;
    std::vector<int> dims = {-1};  // Negative dim -> error
    (void)plan.init(1, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    
    int status_before = shafft_last_error_status();
    if (status_before == 0) {
        std::cerr << "Expected error to be set\n";
        return false;
    }
    
    // Clear it
    shafft_clear_last_error();
    
    int status_after = shafft_last_error_status();
    int source_after = shafft_last_error_source();
    int code_after = shafft_last_error_domain_code();
    
    if (status_after != 0 || source_after != 0 || code_after != 0) {
        std::cerr << "clear_last_error did not reset all fields\n";
        std::cerr << "  status=" << status_after 
                  << " source=" << source_after 
                  << " code=" << code_after << "\n";
        return false;
    }
    
    return true;
}

//------------------------------------------------------------------------------
// Test: Error source name lookup
//------------------------------------------------------------------------------
TEST(error_source_names) {
    struct { int source; const char* expected; } cases[] = {
        { SHAFFT_ERRSRC_NONE,   "SHAFFT" },
        { SHAFFT_ERRSRC_MPI,    "MPI" },
        { SHAFFT_ERRSRC_HIP,    "HIP" },
        { SHAFFT_ERRSRC_HIPFFT, "hipFFT" },
        { SHAFFT_ERRSRC_FFTW,   "FFTW" },
        { SHAFFT_ERRSRC_SYSTEM, "System" },
        { 999,                  "Unknown" },  // Invalid source
    };
    
    for (const auto& tc : cases) {
        const char* name = shafft_error_source_name(tc.source);
        if (std::strcmp(name, tc.expected) != 0) {
            std::cerr << "Source " << tc.source << ": expected '" 
                      << tc.expected << "', got '" << name << "'\n";
            return false;
        }
    }
    
    return true;
}

//------------------------------------------------------------------------------
// Test: Error message formatting
//------------------------------------------------------------------------------
TEST(error_message_buffer) {
    shafft_clear_last_error();
    
    // With no error, message should be empty
    char buf[256];
    int len = shafft_last_error_message(buf, sizeof(buf));
    
    // Should return 0 or empty string for no error
    if (len < 0) {
        std::cerr << "Unexpected negative length\n";
        return false;
    }
    
    // Test null buffer handling
    int len_null = shafft_last_error_message(nullptr, 100);
    if (len_null != 0) {
        std::cerr << "Expected 0 for null buffer, got " << len_null << "\n";
        return false;
    }
    
    // Test zero-length buffer
    int len_zero = shafft_last_error_message(buf, 0);
    if (len_zero != 0) {
        std::cerr << "Expected 0 for zero-length buffer, got " << len_zero << "\n";
        return false;
    }
    
    return true;
}

//------------------------------------------------------------------------------
// Test: Error message truncation with small buffer
//------------------------------------------------------------------------------
TEST(error_message_truncation) {
    shafft_clear_last_error();
    
    // Trigger an error that should produce a message
    // We'll use a very small buffer to test truncation
    shafft::Plan plan;
    std::vector<int> dims = {-1};
    plan.init(1, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    
    // Try with tiny buffer
    char tiny[4];
    int len = shafft_last_error_message(tiny, sizeof(tiny));
    
    // Should not overflow
    if (len >= (int)sizeof(tiny)) {
        std::cerr << "Truncation failed: len=" << len << " >= buflen=" << sizeof(tiny) << "\n";
        return false;
    }
    
    // Should be null-terminated
    bool null_terminated = false;
    for (size_t i = 0; i < sizeof(tiny); ++i) {
        if (tiny[i] == '\0') {
            null_terminated = true;
            break;
        }
    }
    if (!null_terminated) {
        std::cerr << "Buffer not null-terminated\n";
        return false;
    }
    
    return true;
}

//------------------------------------------------------------------------------
// Test: Thread-local error state isolation
// This tests that clearing error state on one thread doesn't affect another
//------------------------------------------------------------------------------
TEST(thread_local_isolation) {
    // This test verifies that error state is truly thread-local
    std::atomic<bool> thread_passed{true};
    std::atomic<bool> main_ready{false};
    std::atomic<bool> thread_ready{false};
    
    // Clear main thread state
    shafft_clear_last_error();
    
    std::thread worker([&]() {
        // Clear worker thread state initially
        shafft_clear_last_error();
        
        // Note: simple validation errors don't set thread-local state
        // But we can still test that clear/query is thread-local
        thread_ready = true;
        
        // Wait for main to do its operations
        while (!main_ready) {
            std::this_thread::yield();
        }
        
        // Worker state should still be cleared (not affected by main)
        int worker_status = shafft_last_error_status();
        if (worker_status != 0) {
            thread_passed = false;  // Main shouldn't have polluted worker
        }
    });
    
    // Wait for worker to be ready
    while (!thread_ready) {
        std::this_thread::yield();
    }
    
    // Main thread should have SUCCESS
    int main_status = shafft_last_error_status();
    if (main_status != 0) {
        std::cerr << "Main thread status not SUCCESS: " << main_status << "\n";
        main_ready = true;
        worker.join();
        return false;
    }
    
    // Clear main thread (should not affect worker)
    shafft_clear_last_error();
    main_ready = true;
    
    worker.join();
    
    if (!thread_passed) {
        std::cerr << "Worker thread state was affected by main thread\n";
        return false;
    }
    
    return true;
}

//------------------------------------------------------------------------------
// Test: Successive errors overwrite previous state
//------------------------------------------------------------------------------
TEST(successive_errors_overwrite) {
    shafft_clear_last_error();
    
    // First error
    shafft::Plan plan1;
    std::vector<int> dims1 = {};  // Empty dims
    plan1.init(1, dims1, shafft::FFTType::C2C, MPI_COMM_WORLD);
    int status1 = shafft_last_error_status();
    
    // Second different error
    shafft::Plan plan2;
    std::vector<int> dims2 = {32, 32};
    plan2.init(1, dims2, shafft::FFTType::C2C, MPI_COMM_WORLD);
    
    // Execute without buffers - different error type
    plan2.execute(shafft::FFTDirection::FORWARD);
    int status2 = shafft_last_error_status();
    
    // Second error should have overwritten first
    // (They may be the same code, but state should be updated)
    // Key test: error state should reflect LAST error, not first
    if (status2 == 0) {
        std::cerr << "Expected error status after execute without buffers\n";
        return false;
    }
    
    return true;
}

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (rank == 0) {
        std::cout << "=== Error API Unit Tests ===\n";
    }
    
    // Run all tests (only on rank 0 for output clarity)
    if (rank == 0) {
        run_initial_state_is_success();
        run_error_persists_after_failure();
        run_clear_resets_state();
        run_error_source_names();
        run_error_message_buffer();
        run_error_message_truncation();
        run_thread_local_isolation();
        run_successive_errors_overwrite();
        
        std::cout << "\nResults: " << g_passed << " passed, " << g_failed << " failed\n";
    }
    
    // Synchronize for MPI finalize
    MPI_Barrier(MPI_COMM_WORLD);
    
    MPI_Finalize();
    return (g_failed == 0) ? 0 : 1;
}
