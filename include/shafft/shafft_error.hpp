/** @file shafft_error.hpp
 *  @brief Error handling utilities and macros for SHAFFT.
 *  @ingroup cpp_api
 */

#ifndef SHAFFT_ERROR_HPP
#define SHAFFT_ERROR_HPP

#include <shafft/shafft.h>        // C ABI: shafft_errsrc_t, error query functions
#include <shafft/shafft_types.hpp>

#include <exception>  // std::exception
#include <new>        // std::bad_alloc

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
  try {
    throw;
  } catch (const std::bad_alloc&) {
    return Status::SHAFFT_ERR_ALLOC;
  } catch (const std::exception&) {
    return Status::SHAFFT_ERR_INTERNAL;
  } catch (...) {
    return Status::SHAFFT_ERR_INTERNAL;
  }
}

}  // namespace err
}  // namespace shafft

// ==================================
// C ABI types and functions are now provided by shafft.h
// ==================================

// Internal C++ setter used by implementation (not part of public C API)
namespace shafft {
namespace detail {
void set_last_error(shafft::Status st, shafft_errsrc_t src, int domain_code) noexcept;
}
}  // namespace shafft

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
#define SHAFFT_CATCH_RETURN()                                       \
  catch (...) {                                                     \
    return static_cast<int>(::shafft::err::mapExceptionToStatus()); \
  }

// Explicit cast helper when you need an int code.
#define SHAFFT_STATUS(code) static_cast<int>(::shafft::Status::code)

// Early-return helpers.
#define SHAFFT_RETURN(code) return static_cast<int>(::shafft::Status::code)

#define SHAFFT_RETURN_IF(cond, code) \
  do {                               \
    if (cond)                        \
      SHAFFT_RETURN(code);           \
  } while (0)

// Fail helper: set last error and return the status as int.
#define SHAFFT_FAIL_WITH(status_enum, source_enum, raw_code)                                    \
  do {                                                                                          \
    ::shafft::detail::set_last_error(::shafft::Status::status_enum, (source_enum), (raw_code)); \
    return static_cast<int>(::shafft::Status::status_enum);                                     \
  } while (0)

// Fail helper for SHAFFT-internal validation errors (no external domain code).
#define SHAFFT_FAIL(status_enum) SHAFFT_FAIL_WITH(status_enum, SHAFFT_ERRSRC_NONE, 0)

// Wrap an MPI call; on failure set last error and return ERR_MPI.
#define SHAFFT_MPI_OR_FAIL(expr)                                      \
  do {                                                                \
    int _mpi_rc__ = (expr);                                           \
    if (_mpi_rc__ != MPI_SUCCESS) {                                   \
      SHAFFT_FAIL_WITH(SHAFFT_ERR_MPI, SHAFFT_ERRSRC_MPI, _mpi_rc__); \
    }                                                                 \
  } while (0)

// Wrap a HIP runtime call; on failure set last error and return ERR_BACKEND.
#define SHAFFT_HIP_CHECK(rc_expr)                                         \
  do {                                                                    \
    int _hip_rc__ = (rc_expr);                                            \
    if (_hip_rc__ != 0) {                                                 \
      SHAFFT_FAIL_WITH(SHAFFT_ERR_BACKEND, SHAFFT_ERRSRC_HIP, _hip_rc__); \
    }                                                                     \
  } while (0)

// Wrap a hipFFT call; on failure set last error and return ERR_BACKEND.
#define SHAFFT_HIPFFT_CHECK(rc_expr)                                         \
  do {                                                                       \
    int _fft_rc__ = (rc_expr);                                               \
    if (_fft_rc__ != 0) {                                                    \
      SHAFFT_FAIL_WITH(SHAFFT_ERR_BACKEND, SHAFFT_ERRSRC_HIPFFT, _fft_rc__); \
    }                                                                        \
  } while (0)

// Wrap an FFTW call; on failure set last error and return ERR_BACKEND.
#define SHAFFT_FFTW_CHECK(rc_expr)                                          \
  do {                                                                      \
    int _fftw_rc__ = (rc_expr);                                             \
    if (_fftw_rc__ != 0) {                                                  \
      SHAFFT_FAIL_WITH(SHAFFT_ERR_BACKEND, SHAFFT_ERRSRC_FFTW, _fftw_rc__); \
    }                                                                       \
  } while (0)

// Generic backend check - uses the appropriate source based on compile-time backend.
// For finer-grained control, use SHAFFT_HIP_CHECK, SHAFFT_HIPFFT_CHECK, or SHAFFT_FFTW_CHECK.
#if SHAFFT_BACKEND_HIPFFT
#define SHAFFT_BACKEND_CHECK(rc_expr) SHAFFT_HIPFFT_CHECK(rc_expr)
#else
#define SHAFFT_BACKEND_CHECK(rc_expr) SHAFFT_FFTW_CHECK(rc_expr)
#endif

#endif  // SHAFFT_ERROR_HPP
