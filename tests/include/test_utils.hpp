/**
 * @file test_utils.hpp
 * @brief Common utilities for SHAFFT tests
 */
#ifndef SHAFFT_TEST_UTILS_HPP
#define SHAFFT_TEST_UTILS_HPP

#include <cmath>
#include <complex>
#include <cstdio>
#include <mpi.h>
#include <shafft/shafft.h>
#include <shafft/shafft.hpp>
#include <shafft/shafft_config.h>
#include <vector>

// Check HIP return codes - returns false on failure (only when HIP backend)
#if SHAFFT_BACKEND_HIPFFT
#include <hip/hip_runtime.h>
#define HIP_CHECK(x)                                                                               \
  do {                                                                                             \
    hipError_t err_ = (x);                                                                         \
    if (err_ != hipSuccess) {                                                                      \
      std::fprintf(                                                                                \
          stderr, "HIP error: %s at %s:%d\n", hipGetErrorString(err_), __FILE__, __LINE__);        \
      return false;                                                                                \
    }                                                                                              \
  } while (0)
#endif

namespace test {

/// Check SHAFFT return code and print error details if failed.
/// Returns false on failure, allowing use in test functions.
inline bool check_shafft(int rc, const char* file, int line) {
  if (rc == 0)
    return true;

  char msg[256];
  shafftLastErrorMessage(msg, sizeof(msg));
  const char* source = shafftErrorSourceName(shafftLastErrorSource());
  int domain_code = shafftLastErrorDomainCode();

  std::fprintf(stderr, "SHAFFT error at %s:%d\n", file, line);
  std::fprintf(stderr, "  Status: %d, Source: %s, Domain code: %d\n", rc, source, domain_code);
  if (msg[0] != '\0') {
    std::fprintf(stderr, "  Message: %s\n", msg);
  }
  return false;
}

/// Macro for convenient SHAFFT error checking with file/line info
#define SHAFFT_CHECK(x)                                                                            \
  if (!test::check_shafft((x), __FILE__, __LINE__))                                                \
  return false

// Base tolerances for float/double comparisons (before size scaling)
// For FFT accuracy checks, use check_rel_error() which applies sqrt(log N) scaling
constexpr float TOL_F = 1e-5f;
constexpr double TOL_D = 1e-12;

// Approximate equality for complex numbers
inline bool approx_eq(shafft::complexf a, shafft::complexf b, float tol = TOL_F) {
  return std::fabs(a.real() - b.real()) < tol && std::fabs(a.imag() - b.imag()) < tol;
}

inline bool approx_eq(shafft::complexd a, shafft::complexd b, double tol = TOL_D) {
  return std::fabs(a.real() - b.real()) < tol && std::fabs(a.imag() - b.imag()) < tol;
}

// Reduce pass/fail across all MPI ranks
// Returns true only if all ranks pass
inline bool all_ranks_pass(bool localPass, MPI_Comm comm = MPI_COMM_WORLD) {
  int local = localPass ? 1 : 0;
  int global = 0;
  MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, comm);
  return global == 1;
}

// Distributed relative error (FFTW-style) between two local slabs.
// Linf = max|a-b| / max|b|,  L2 = ||a-b||2 / ||b||2, scaled by sqrt(log N).
// Returns true if both norms are below tol_rel * sqrt(log N).
template <typename ComplexT>
inline bool check_rel_error(const ComplexT* out,
                            const ComplexT* ref,
                            size_t local_count,
                            size_t globalN,
                            MPI_Comm comm,
                            double base_tol_rel) {
  double localMaxErr = 0.0;
  double local_max_ref = 0.0;
  double local_err2 = 0.0;
  double local_ref2 = 0.0;

  for (size_t i = 0; i < local_count; ++i) {
    double er = static_cast<double>(out[i].real()) - static_cast<double>(ref[i].real());
    double ei = static_cast<double>(out[i].imag()) - static_cast<double>(ref[i].imag());
    double err_mag = std::hypot(er, ei);
    double rr = static_cast<double>(ref[i].real());
    double ri = static_cast<double>(ref[i].imag());
    double ref_mag = std::hypot(rr, ri);
    localMaxErr = std::max(localMaxErr, err_mag);
    local_max_ref = std::max(local_max_ref, ref_mag);
    local_err2 += err_mag * err_mag;
    local_ref2 += ref_mag * ref_mag;
  }

  double global_max_err = 0.0, global_max_ref = 0.0;
  double global_err2 = 0.0, global_ref2 = 0.0;
  MPI_Allreduce(&localMaxErr, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, comm);
  MPI_Allreduce(&local_max_ref, &global_max_ref, 1, MPI_DOUBLE, MPI_MAX, comm);
  MPI_Allreduce(&local_err2, &global_err2, 1, MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(&local_ref2, &global_ref2, 1, MPI_DOUBLE, MPI_SUM, comm);

  double rel_linf = (global_max_ref > 0.0) ? global_max_err / global_max_ref : global_max_err;
  double rel_l2 =
      (global_ref2 > 0.0) ? std::sqrt(global_err2 / global_ref2) : std::sqrt(global_err2);

  double scale = std::sqrt(std::log(static_cast<double>(std::max<size_t>(globalN, 2))));
  double tol = base_tol_rel * scale;
  return (rel_linf <= tol) && (rel_l2 <= tol);
}

// Overload taking vectors (uses min of both sizes as local_count)
template <typename ComplexT>
inline bool check_rel_error(const std::vector<ComplexT>& out,
                            const std::vector<ComplexT>& ref,
                            size_t globalN,
                            MPI_Comm comm,
                            double base_tol_rel) {
  size_t n = std::min(out.size(), ref.size());
  return check_rel_error(out.data(), ref.data(), n, globalN, comm, base_tol_rel);
}

// Print only on rank 0
template <typename... Args>
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

/**
 * @brief Simple test runner for unit tests (single-rank, no MPI sync)
 *
 * Usage:
 *   test::TestRunner runner("My Unit Tests");
 *   runner.run("test_name", test_function);
 *   return runner.finalize();
 */
class TestRunner {
public:
  TestRunner(const char* suite_name) : suite_name_(suite_name), all_passed_(true) {
    std::printf("=== %s ===\n", suite_name_);
    std::printf("SHAFFT %s (backend: %s)\n", shafft::getVersionString(), shafft::getBackendName());
    std::printf("\n");
  }

  void run(const char* test_name, bool (*test_fn)()) {
    bool passed = test_fn();
    std::printf("  %-40s %s\n", test_name, passed ? "PASS" : "FAIL");

    if (!passed)
      all_passed_ = false;
    results_.push_back({test_name, passed});
  }

  int finalize() {
    int passed = 0, failed = 0;
    for (const auto& r : results_) {
      if (r.passed)
        passed++;
      else
        failed++;
    }
    std::printf("\n=== %s: %s ===\n", suite_name_, all_passed_ ? "ALL PASSED" : "FAILED");
    std::printf("Results: %d passed, %d failed\n", passed, failed);
    return all_passed_ ? 0 : 1;
  }

private:
  const char* suite_name_;
  bool all_passed_;
  std::vector<TestResult> results_;
};

/**
 * @brief MPI-aware test runner for integration tests (multi-rank, with MPI sync)
 *
 * Synchronizes pass/fail across all MPI ranks using all_ranks_pass().
 * Only rank 0 prints output.
 *
 * Usage:
 *   test::TestRunnerMPI runner("My Integration Tests");
 *   runner.run("test_name", test_function);
 *   return runner.finalize();
 */
class TestRunnerMPI {
public:
  TestRunnerMPI(const char* suite_name) : suite_name_(suite_name), all_passed_(true) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    rank_ = rank;

    if (rank_ == 0) {
      std::printf("=== %s ===\n", suite_name_);
      std::printf(
          "SHAFFT %s (backend: %s)\n", shafft::getVersionString(), shafft::getBackendName());
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

    if (!passed)
      all_passed_ = false;
    results_.push_back({test_name, passed});
  }

  int finalize() {
    if (rank_ == 0) {
      std::printf("\n=== %s: %s ===\n", suite_name_, all_passed_ ? "ALL PASSED" : "FAILED");
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
  for (int x : v)
    p *= static_cast<size_t>(x);
  return p;
}

inline size_t product(const std::vector<size_t>& v) {
  size_t p = 1;
  for (size_t x : v)
    p *= x;
  return p;
}

// Compute commDims for nda=1 (slab) decomposition
// Decomposes only the first axis up to min(worldSize, dims[0])
inline std::vector<int> compute_comm_dims_nda1(const std::vector<size_t>& dims, MPI_Comm comm) {
  int worldSize;
  MPI_Comm_size(comm, &worldSize);
  std::vector<int> commDims(dims.size(), 1);
  commDims[0] = std::min(worldSize, static_cast<int>(dims[0]));
  return commDims;
}

// Compute max absolute error between two complex arrays
template <typename T>
double max_error(const T* a, const T* b, size_t n) {
  double max_err = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double err_re = std::fabs(a[i].real() - b[i].real());
    double err_im = std::fabs(a[i].imag() - b[i].imag());
    max_err = std::max(max_err, std::max(err_re, err_im));
  }
  return max_err;
}

// Generate deterministic test input (sin/cos pattern)
template <typename Complex = std::complex<double>>
std::vector<Complex> make_input(size_t n) {
  using Real = typename Complex::value_type;
  std::vector<Complex> v(n);
  for (size_t i = 0; i < n; ++i) {
    v[i] = Complex(static_cast<Real>(std::sin(0.01 * i)), static_cast<Real>(std::cos(0.02 * i)));
  }
  return v;
}

} // namespace test

#endif // SHAFFT_TEST_UTILS_HPP
