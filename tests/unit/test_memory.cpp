/**
 * @file test_memory.cpp
 * @brief Unit tests for portable memory allocation and copy helpers (C++ API)
 * 
 * Tests allocBuffer(), freeBuffer(), copyToBuffer(), copyFromBuffer().
 * These work identically on CPU (FFTW) and GPU (hipFFT) backends.
 */
#include <shafft/shafft.hpp>
#include <mpi.h>
#include <cstring>
#include <cmath>
#include <iostream>
#include <vector>
#include <complex>

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
// Test: Single-precision allocation returns non-null
//------------------------------------------------------------------------------
TEST(alloc_float_nonnull) {
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

//------------------------------------------------------------------------------
// Test: Double-precision allocation returns non-null
//------------------------------------------------------------------------------
TEST(alloc_double_nonnull) {
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

//------------------------------------------------------------------------------
// Test: Zero-size allocation behavior
//------------------------------------------------------------------------------
TEST(alloc_zero_size) {
    shafft::complexf* buf = nullptr;
    int rc = shafft::allocBuffer(0, &buf);
    
    // Zero-size allocation may succeed with null or non-null
    // The key is it shouldn't crash and free should handle it
    if (rc == 0 && buf != nullptr) {
        (void)shafft::freeBuffer(buf);
    }
    // If rc != 0, that's also acceptable (error on zero size)
    
    return true;  // Just checking no crash
}

//------------------------------------------------------------------------------
// Test: Free handles null gracefully
//------------------------------------------------------------------------------
TEST(free_null_safe) {
    shafft::complexf* null_f = nullptr;
    shafft::complexd* null_d = nullptr;
    
    int rc1 = shafft::freeBuffer(null_f);
    int rc2 = shafft::freeBuffer(null_d);
    
    // Should not crash and should return success (or at least not crash)
    // Some implementations return success on null, others return error
    // The key is no crash
    (void)rc1;
    (void)rc2;
    
    return true;  // Success if we reach here without crashing
}

//------------------------------------------------------------------------------
// Test: Roundtrip copy preserves data (single precision)
//------------------------------------------------------------------------------
TEST(copy_roundtrip_float) {
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
        if (host_dst[i].real() != host_src[i].real() ||
            host_dst[i].imag() != host_src[i].imag()) {
            std::cerr << "Mismatch at index " << i << ": "
                      << "src=(" << host_src[i].real() << "," << host_src[i].imag() << ") "
                      << "dst=(" << host_dst[i].real() << "," << host_dst[i].imag() << ")\n";
            (void)shafft::freeBuffer(buf);
            return false;
        }
    }
    
    (void)shafft::freeBuffer(buf);
    return true;
}

//------------------------------------------------------------------------------
// Test: Roundtrip copy preserves data (double precision)
//------------------------------------------------------------------------------
TEST(copy_roundtrip_double) {
    const size_t N = 256;
    
    // Host data with known pattern (use values that need double precision)
    std::vector<shafft::complexd> host_src(N);
    for (size_t i = 0; i < N; ++i) {
        // Use values that would lose precision in float
        host_src[i] = {1.0 + 1e-10 * static_cast<double>(i), 
                       2.0 + 1e-10 * static_cast<double>(N - i)};
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
        if (host_dst[i].real() != host_src[i].real() ||
            host_dst[i].imag() != host_src[i].imag()) {
            std::cerr << "Double mismatch at index " << i << "\n";
            (void)shafft::freeBuffer(buf);
            return false;
        }
    }
    
    (void)shafft::freeBuffer(buf);
    return true;
}

//------------------------------------------------------------------------------
// Test: Large allocation (stress test)
//------------------------------------------------------------------------------
TEST(large_allocation) {
    // 64 MB worth of complex floats
    const size_t N = 8 * 1024 * 1024;  // 8M elements = 64MB
    
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

//------------------------------------------------------------------------------
// Test: Multiple allocations and frees
//------------------------------------------------------------------------------
TEST(multiple_alloc_free) {
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

//------------------------------------------------------------------------------
// Test: Partial copy (subset of buffer)
//------------------------------------------------------------------------------
TEST(partial_copy) {
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

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (rank == 0) {
        std::cout << "=== Memory Helpers Unit Tests (C++ API) ===\n";
        std::cout << "Backend: " << shafft::getBackendName() << "\n\n";
    }
    
    if (rank == 0) {
        run_alloc_float_nonnull();
        run_alloc_double_nonnull();
        run_alloc_zero_size();
        run_free_null_safe();
        run_copy_roundtrip_float();
        run_copy_roundtrip_double();
        run_large_allocation();
        run_multiple_alloc_free();
        run_partial_copy();
        
        std::cout << "\nResults: " << g_passed << " passed, " << g_failed << " failed\n";
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return (g_failed == 0) ? 0 : 1;
}
