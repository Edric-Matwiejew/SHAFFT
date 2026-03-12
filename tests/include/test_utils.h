/**
 * @file test_utils.h
 * @brief Common utilities for SHAFFT C tests
 *
 * Mirrors the C++ test_utils.hpp for consistent validation standards.
 */
#ifndef SHAFFT_TEST_UTILS_H
#define SHAFFT_TEST_UTILS_H

#include <math.h>
#include <mpi.h>
#include <shafft/shafft.h>
#include <stdarg.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Base tolerances (before size scaling) - matches C++ test_utils.hpp */
#define TOL_F 1e-5f
#define TOL_D 1e-12

/* Single-precision complex type (matches shafft's layout) */
typedef struct {
  float real;
  float imag;
} test_complexf;

/* Double-precision complex type */
typedef struct {
  double real;
  double imag;
} test_complexd;

/**
 * @brief Check SHAFFT return code and print error details if failed.
 * @param rc Return code from SHAFFT function
 * @param label Description of the operation
 * @param file Source file name (__FILE__)
 * @param line Source line number (__LINE__)
 * @return 1 if error, 0 if success
 */
static inline int check_shafft_c(int rc, const char* label, const char* file, int line) {
  if (rc == 0)
    return 0;
  char msg[256] = {0};
  shafftLastErrorMessage(msg, sizeof(msg));
  const char* source = shafftErrorSourceName(shafftLastErrorSource());
  int domain_code = shafftLastErrorDomainCode();
  fprintf(stderr, "SHAFFT error at %s:%d\n", file, line);
  fprintf(
      stderr, "  %s failed: rc=%d, source=%s, domain_code=%d\n", label, rc, source, domain_code);
  if (msg[0] != '\0') {
    fprintf(stderr, "  Message: %s\n", msg);
  }
  return 1;
}

#define SHAFFT_CHECK_C(expr, label)                                                                \
  do {                                                                                             \
    if (check_shafft_c((expr), (label), __FILE__, __LINE__))                                       \
      return 1;                                                                                    \
  } while (0)

/**
 * @brief Reduce pass/fail across all MPI ranks.
 * @param localPass 1 if this rank passes, 0 if fails
 * @param comm MPI communicator
 * @return 1 only if ALL ranks pass
 */
static inline int all_ranks_pass_c(int localPass, MPI_Comm comm) {
  int global = 0;
  MPI_Allreduce(&localPass, &global, 1, MPI_INT, MPI_MIN, comm);
  return global;
}

/**
 * @brief Compute FFTW-style relative error for single precision.
 *
 * Computes both L∞ and L2 relative errors across all MPI ranks,
 * scaled by sqrt(log N) per FFTW methodology.
 *
 * @param out Output data (result)
 * @param ref Reference data (original)
 * @param local_count Number of local elements
 * @param globalN Total global element count (for scaling)
 * @param comm MPI communicator
 * @param base_tol Base tolerance (before scaling)
 * @return 1 if error is within tolerance, 0 if exceeds
 */
static inline int check_rel_error_f(const test_complexf* out,
                                    const test_complexf* ref,
                                    size_t local_count,
                                    size_t globalN,
                                    MPI_Comm comm,
                                    double base_tol) {
  double localMaxErr = 0.0;
  double local_max_ref = 0.0;
  double local_err2 = 0.0;
  double local_ref2 = 0.0;

  for (size_t i = 0; i < local_count; ++i) {
    double er = (double)out[i].real - (double)ref[i].real;
    double ei = (double)out[i].imag - (double)ref[i].imag;
    double err_mag = sqrt(er * er + ei * ei);
    double rr = (double)ref[i].real;
    double ri = (double)ref[i].imag;
    double ref_mag = sqrt(rr * rr + ri * ri);

    if (err_mag > localMaxErr)
      localMaxErr = err_mag;
    if (ref_mag > local_max_ref)
      local_max_ref = ref_mag;
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
  double rel_l2 = (global_ref2 > 0.0) ? sqrt(global_err2 / global_ref2) : sqrt(global_err2);

  /* Scale tolerance by sqrt(log N) per FFTW methodology */
  double scale = sqrt(log((double)(globalN > 2 ? globalN : 2)));
  double tol = base_tol * scale;

  return (rel_linf <= tol) && (rel_l2 <= tol);
}

/**
 * @brief Compute FFTW-style relative error for double precision.
 */
static inline int check_rel_error_d(const test_complexd* out,
                                    const test_complexd* ref,
                                    size_t local_count,
                                    size_t globalN,
                                    MPI_Comm comm,
                                    double base_tol) {
  double localMaxErr = 0.0;
  double local_max_ref = 0.0;
  double local_err2 = 0.0;
  double local_ref2 = 0.0;

  for (size_t i = 0; i < local_count; ++i) {
    double er = out[i].real - ref[i].real;
    double ei = out[i].imag - ref[i].imag;
    double err_mag = sqrt(er * er + ei * ei);
    double ref_mag = sqrt(ref[i].real * ref[i].real + ref[i].imag * ref[i].imag);

    if (err_mag > localMaxErr)
      localMaxErr = err_mag;
    if (ref_mag > local_max_ref)
      local_max_ref = ref_mag;
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
  double rel_l2 = (global_ref2 > 0.0) ? sqrt(global_err2 / global_ref2) : sqrt(global_err2);

  double scale = sqrt(log((double)(globalN > 2 ? globalN : 2)));
  double tol = base_tol * scale;

  return (rel_linf <= tol) && (rel_l2 <= tol);
}

/**
 * @brief Print only on rank 0.
 */
static inline void print0(const char* fmt, ...) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
  }
}

/**
 * @brief Compute product of integer array elements.
 */
static inline size_t product_i(const int* arr, int n) {
  size_t p = 1;
  for (int i = 0; i < n; ++i)
    p *= (size_t)arr[i];
  return p;
}

#ifdef __cplusplus
}
#endif

#endif /* SHAFFT_TEST_UTILS_H */
