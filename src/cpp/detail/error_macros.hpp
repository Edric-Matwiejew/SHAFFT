// error_macros.hpp - Internal error handling macros
// Not for use by library consumers.

#ifndef SHAFFT_SRC_ERROR_MACROS_HPP
#define SHAFFT_SRC_ERROR_MACROS_HPP

#include <shafft/shafft.h>
#include <shafft/shafft_error.hpp>
#include <shafft/shafft_types.hpp>

namespace shafft::detail {

// Set thread-local error state.
void setLastError(shafft::Status st, shafft_errsrc_t src, int domainCode) noexcept;

} // namespace shafft::detail

// Catch any exception and return mapped status code.
#define SHAFFT_CATCH_RETURN()                                                                      \
  catch (...) {                                                                                    \
    return static_cast<int>(::shafft::err::mapExceptionToStatus());                                \
  }

// Return a status code as integer.
#define SHAFFT_STATUS(code) static_cast<int>(code)

// Set error state and return status code.
#define SHAFFT_FAIL_WITH(status_enum, source_enum, raw_code)                                       \
  do {                                                                                             \
    ::shafft::detail::setLastError(                                                                \
        static_cast<::shafft::Status>(status_enum), (source_enum), (raw_code));                    \
    return static_cast<int>(status_enum);                                                          \
  } while (0)

// Set error state and return status code (no domain code).
#define SHAFFT_FAIL(status_enum) SHAFFT_FAIL_WITH(status_enum, SHAFFT_ERRSRC_NONE, 0)

// Check MPI call and fail if it fails.
#define SHAFFT_MPI_OR_FAIL(expr)                                                                   \
  do {                                                                                             \
    int _mpi_rc__ = (expr);                                                                        \
    if (_mpi_rc__ != MPI_SUCCESS) {                                                                \
      SHAFFT_FAIL_WITH(SHAFFT_ERR_MPI, SHAFFT_ERRSRC_MPI, _mpi_rc__);                              \
    }                                                                                              \
  } while (0)

// Check HIP runtime call and fail if it fails.
#define SHAFFT_HIP_CHECK(rc_expr)                                                                  \
  do {                                                                                             \
    hipError_t _hip_rc__ = (rc_expr);                                                              \
    if (_hip_rc__ != hipSuccess) {                                                                 \
      SHAFFT_FAIL_WITH(SHAFFT_ERR_BACKEND, SHAFFT_ERRSRC_HIP, static_cast<int>(_hip_rc__));        \
    }                                                                                              \
  } while (0)

// Check hipFFT call and fail if it fails.
#define SHAFFT_HIPFFT_CHECK(rc_expr)                                                               \
  do {                                                                                             \
    int _fft_rc__ = (rc_expr);                                                                     \
    if (_fft_rc__ != 0) {                                                                          \
      SHAFFT_FAIL_WITH(SHAFFT_ERR_BACKEND, SHAFFT_ERRSRC_HIPFFT, _fft_rc__);                       \
    }                                                                                              \
  } while (0)

// Check FFTW call and fail with FFTW error source if it fails.
#define SHAFFT_FFTW_CHECK(rc_expr)                                                                 \
  do {                                                                                             \
    int _fftw_rc__ = (rc_expr);                                                                    \
    if (_fftw_rc__ != 0) {                                                                         \
      SHAFFT_FAIL_WITH(SHAFFT_ERR_BACKEND, SHAFFT_ERRSRC_FFTW, _fftw_rc__);                        \
    }                                                                                              \
  } while (0)

// Backend-agnostic FFT check macro.
#if SHAFFT_BACKEND_HIPFFT
#define SHAFFT_BACKEND_CHECK(rc_expr) SHAFFT_HIPFFT_CHECK(rc_expr)
#else
#define SHAFFT_BACKEND_CHECK(rc_expr) SHAFFT_FFTW_CHECK(rc_expr)
#endif

#endif // SHAFFT_SRC_ERROR_MACROS_HPP
