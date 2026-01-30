/** @file shafft_error.hpp
 *  @brief Error handling utilities and macros for SHAFFT.
 *  @ingroup cpp_api
 */

#ifndef SHAFFT_ERROR_HPP
#define SHAFFT_ERROR_HPP

#include <new>         // std::bad_alloc
#include <exception>   // std::exception
#include <mpi.h>
#include <shafft/shafft_types.hpp>

// ======================
// Public status helpers
// ======================
namespace shafft {
namespace err {

/**
 * @brief Map any active exception to a Status code (use only in a catch block).
 * @ingroup cpp_api
 */
[[nodiscard]] inline Status mapExceptionToStatus() noexcept {
  try { throw; }
  catch (const std::bad_alloc&) { return Status::SHAFFT_ERR_ALLOC; }
  catch (const std::exception&) { return Status::SHAFFT_ERR_INTERNAL; }
  catch (...)                   { return Status::SHAFFT_ERR_INTERNAL; }
}

} // namespace err
} // namespace shafft

// ==================================
// C ABI: thread-local "last error"
// ==================================
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Error source domain for detailed diagnostics.
 *
 * When a SHAFFT function returns an error, call shafft_last_error_source()
 * to determine which subsystem caused the failure, then use
 * shafft_last_error_domain_code() to get the raw error code from that subsystem.
 */
typedef enum {
  SHAFFT_ERRSRC_NONE    = 0,  /**< No error or SHAFFT-internal error */
  SHAFFT_ERRSRC_MPI     = 1,  /**< MPI library error (use MPI_Error_string) */
  SHAFFT_ERRSRC_HIP     = 2,  /**< HIP runtime error (hipError_t) */
  SHAFFT_ERRSRC_HIPFFT  = 3,  /**< hipFFT library error (hipfftResult_t) */
  SHAFFT_ERRSRC_FFTW    = 4,  /**< FFTW library error */
  SHAFFT_ERRSRC_SYSTEM  = 5   /**< OS / allocation / errno-like errors */
} shafft_errsrc_t;

/** @brief Get the SHAFFT status code from the last error. */
int shafft_last_error_status(void);

/** @brief Get the error source domain from the last error. */
int shafft_last_error_source(void);

/** @brief Get the raw domain-specific error code from the last error. */
int shafft_last_error_domain_code(void);

/**
 * @brief Get a human-readable message for the last error.
 * @param buf    Buffer to receive the message.
 * @param buflen Size of the buffer.
 * @return Number of characters written (excluding null terminator).
 */
int shafft_last_error_message(char* buf, int buflen);

/** @brief Clear the last error state. */
void shafft_clear_last_error(void);

/**
 * @brief Get the name of an error source as a string.
 * @param source Error source value from shafft_last_error_source().
 * @return String name of the error source (e.g., "MPI", "HIP", "hipFFT").
 */
const char* shafft_error_source_name(int source);

#ifdef __cplusplus
} /* extern "C" */
#endif

// Internal C++ setter used by implementation (not part of public C API)
#ifdef __cplusplus
namespace shafft { namespace detail {
  void set_last_error(shafft::Status st, shafft_errsrc_t src, int domain_code) noexcept;
}}
#endif

// =========================
// Convenience macros (int)
// =========================

// Translate any exception to a Status code at API boundaries.
// Usage:
//   int api() noexcept {
//     try {
//       ...
//       return SHAFFT_STATUS(SHAFFT_SUCCESS);
//     }
//     SHAFFT_CATCH_RETURN();
//   }
#define SHAFFT_CATCH_RETURN() \
  catch (...) { return static_cast<int>(::shafft::err::mapExceptionToStatus()); }

// Explicit cast helper when you need an int code.
#define SHAFFT_STATUS(code) static_cast<int>(::shafft::Status::code)

// Early-return helpers.
#define SHAFFT_RETURN(code) \
  return static_cast<int>(::shafft::Status::code)

#define SHAFFT_RETURN_IF(cond, code) \
  do { if (cond) SHAFFT_RETURN(code); } while (0)

// Fail helper: set last error and return the status as int.
#define SHAFFT_FAIL_WITH(status_enum, source_enum, raw_code) \
  do { ::shafft::detail::set_last_error(::shafft::Status::status_enum, \
                                        (source_enum), (raw_code)); \
       return static_cast<int>(::shafft::Status::status_enum); } while (0)

// Fail helper for SHAFFT-internal validation errors (no external domain code).
#define SHAFFT_FAIL(status_enum) \
  SHAFFT_FAIL_WITH(status_enum, SHAFFT_ERRSRC_NONE, 0)

// Wrap an MPI call; on failure set last error and return ERR_MPI.
#define SHAFFT_MPI_OR_FAIL(expr) \
  do { int _mpi_rc__ = (expr); \
       if (_mpi_rc__ != MPI_SUCCESS) { \
         SHAFFT_FAIL_WITH(SHAFFT_ERR_MPI, SHAFFT_ERRSRC_MPI, _mpi_rc__); \
       } } while (0)

// Wrap a HIP runtime call; on failure set last error and return ERR_BACKEND.
#define SHAFFT_HIP_CHECK(rc_expr) \
  do { int _hip_rc__ = (rc_expr); \
       if (_hip_rc__ != 0) { \
         SHAFFT_FAIL_WITH(SHAFFT_ERR_BACKEND, SHAFFT_ERRSRC_HIP, _hip_rc__); \
       } } while (0)

// Wrap a hipFFT call; on failure set last error and return ERR_BACKEND.
#define SHAFFT_HIPFFT_CHECK(rc_expr) \
  do { int _fft_rc__ = (rc_expr); \
       if (_fft_rc__ != 0) { \
         SHAFFT_FAIL_WITH(SHAFFT_ERR_BACKEND, SHAFFT_ERRSRC_HIPFFT, _fft_rc__); \
       } } while (0)

// Wrap an FFTW call; on failure set last error and return ERR_BACKEND.
#define SHAFFT_FFTW_CHECK(rc_expr) \
  do { int _fftw_rc__ = (rc_expr); \
       if (_fftw_rc__ != 0) { \
         SHAFFT_FAIL_WITH(SHAFFT_ERR_BACKEND, SHAFFT_ERRSRC_FFTW, _fftw_rc__); \
       } } while (0)

// Generic backend check - uses the appropriate source based on compile-time backend.
// For finer-grained control, use SHAFFT_HIP_CHECK, SHAFFT_HIPFFT_CHECK, or SHAFFT_FFTW_CHECK.
#if SHAFFT_BACKEND_HIPFFT
#define SHAFFT_BACKEND_CHECK(rc_expr) SHAFFT_HIPFFT_CHECK(rc_expr)
#else
#define SHAFFT_BACKEND_CHECK(rc_expr) SHAFFT_FFTW_CHECK(rc_expr)
#endif

#endif // SHAFFT_ERROR_HPP
