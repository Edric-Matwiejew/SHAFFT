/**
 * @file test_library_info.cpp
 * @brief Unit tests for library information functions (C++ API)
 * 
 * Tests getVersion(), getVersionString(), getBackendName() which are
 * compile-time constants. True unit tests - no MPI communication.
 */
#include <shafft/shafft.hpp>
#include <mpi.h>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <sstream>

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
// Test: C++ Version struct has valid components
//------------------------------------------------------------------------------
TEST(cpp_version_valid) {
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
        std::cerr << "Suspiciously large version: " 
                  << v.major << "." << v.minor << "." << v.patch << "\n";
        return false;
    }
    
    return true;
}

//------------------------------------------------------------------------------
// Test: Version string matches version components
//------------------------------------------------------------------------------
TEST(version_string_matches_components) {
    shafft::Version v = shafft::getVersion();
    const char* vstr = shafft::getVersionString();
    
    if (!vstr) {
        std::cerr << "getVersionString() returned null\n";
        return false;
    }
    
    // Build expected string
    char expected[64];
    snprintf(expected, sizeof(expected), "%d.%d.%d", v.major, v.minor, v.patch);
    
    if (std::strcmp(vstr, expected) != 0) {
        std::cerr << "Version mismatch: getVersionString()='" << vstr 
                  << "' but components give '" << expected << "'\n";
        return false;
    }
    
    return true;
}

//------------------------------------------------------------------------------
// Test: Backend name is valid
//------------------------------------------------------------------------------
TEST(backend_name_valid) {
    const char* name = shafft::getBackendName();
    
    if (!name) {
        std::cerr << "getBackendName() returned null\n";
        return false;
    }
    
    // Should be one of the known backends
    bool valid = (std::strcmp(name, "FFTW") == 0) ||
                 (std::strcmp(name, "hipFFT") == 0);
    
    if (!valid) {
        std::cerr << "Unknown backend name: '" << name << "'\n";
        std::cerr << "Expected 'FFTW' or 'hipFFT'\n";
        return false;
    }
    
    return true;
}

//------------------------------------------------------------------------------
// Test: Version string is stable across calls
//------------------------------------------------------------------------------
TEST(version_string_stable) {
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

//------------------------------------------------------------------------------
// Test: Backend name is stable across calls
//------------------------------------------------------------------------------
TEST(backend_name_stable) {
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

//------------------------------------------------------------------------------
// Test: Version string format is valid (X.Y.Z)
//------------------------------------------------------------------------------
TEST(version_string_format) {
    const char* vstr = shafft::getVersionString();
    
    int major, minor, patch;
    int parsed = sscanf(vstr, "%d.%d.%d", &major, &minor, &patch);
    
    if (parsed != 3) {
        std::cerr << "Version string '" << vstr << "' does not match X.Y.Z format\n";
        return false;
    }
    
    // Rebuild and compare to ensure no extra content
    char rebuilt[64];
    snprintf(rebuilt, sizeof(rebuilt), "%d.%d.%d", major, minor, patch);
    
    if (std::strcmp(vstr, rebuilt) != 0) {
        std::cerr << "Version string has extra content: '" << vstr 
                  << "' vs rebuilt '" << rebuilt << "'\n";
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
        std::cout << "=== Library Info Unit Tests (C++ API) ===\n";
        std::cout << "Backend: " << shafft::getBackendName() << "\n";
        std::cout << "Version: " << shafft::getVersionString() << "\n\n";
    }
    
    if (rank == 0) {
        run_cpp_version_valid();
        run_version_string_matches_components();
        run_backend_name_valid();
        run_version_string_stable();
        run_backend_name_stable();
        run_version_string_format();
        
        std::cout << "\nResults: " << g_passed << " passed, " << g_failed << " failed\n";
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return (g_failed == 0) ? 0 : 1;
}
