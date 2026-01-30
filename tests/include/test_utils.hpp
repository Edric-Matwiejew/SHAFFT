/**
 * @file test_utils.hpp
 * @brief Common utilities for SHAFFT tests
 */
#pragma once

#include <shafft/shafft.hpp>
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <vector>

namespace test {

// Tolerance for float/double comparisons
constexpr float  TOL_F = 1e-5f;
constexpr double TOL_D = 1e-12;

// Approximate equality for complex numbers
inline bool approx_eq(shafft::complexf a, shafft::complexf b, float tol = TOL_F) {
    return std::fabs(a.real() - b.real()) < tol &&
           std::fabs(a.imag() - b.imag()) < tol;
}

inline bool approx_eq(shafft::complexd a, shafft::complexd b, double tol = TOL_D) {
    return std::fabs(a.real() - b.real()) < tol &&
           std::fabs(a.imag() - b.imag()) < tol;
}

// Reduce pass/fail across all MPI ranks
// Returns true only if all ranks pass
inline bool all_ranks_pass(bool local_pass, MPI_Comm comm = MPI_COMM_WORLD) {
    int local = local_pass ? 1 : 0;
    int global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, comm);
    return global == 1;
}

// Print only on rank 0
template<typename... Args>
inline void print0(const char* fmt, Args... args) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::printf(fmt, args...);
    }
}

// Test result tracking
struct TestResult {
    const char* name;
    bool passed;
};

class TestRunner {
public:
    TestRunner(const char* suite_name) : suite_name_(suite_name), all_passed_(true) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        rank_ = rank;
        
        if (rank_ == 0) {
            std::printf("=== %s ===\n", suite_name_);
            std::printf("SHAFFT %s (backend: %s)\n",
                        shafft::getVersionString(), shafft::getBackendName());
            int size;
            MPI_Comm_size(MPI_COMM_WORLD, &size);
            std::printf("MPI ranks: %d\n\n", size);
        }
    }
    
    void run(const char* test_name, bool (*test_fn)()) {
        bool passed = test_fn();
        passed = all_ranks_pass(passed);
        
        if (rank_ == 0) {
            std::printf("  %-40s %s\n", test_name, passed ? "PASS" : "FAIL");
        }
        
        if (!passed) all_passed_ = false;
        results_.push_back({test_name, passed});
    }
    
    int finalize() {
        if (rank_ == 0) {
            std::printf("\n=== %s: %s ===\n", 
                        suite_name_, all_passed_ ? "ALL PASSED" : "FAILED");
        }
        return all_passed_ ? 0 : 1;
    }
    
private:
    const char* suite_name_;
    int rank_;
    bool all_passed_;
    std::vector<TestResult> results_;
};

// Compute product of vector elements
inline size_t product(const std::vector<int>& v) {
    size_t p = 1;
    for (int x : v) p *= static_cast<size_t>(x);
    return p;
}

} // namespace test
