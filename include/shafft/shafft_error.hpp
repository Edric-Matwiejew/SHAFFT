/**
 *  @brief Public error handling utilities for SHAFFT.
 *  @ingroup cpp_raii_api
 */

#ifndef SHAFFT_ERROR_HPP
#define SHAFFT_ERROR_HPP

#include <exception>
#include <new>

#include <shafft/shafft_types.hpp>

namespace shafft::err {

/**
 * @brief Map any active exception in a catch block to a Status code.
 * @ingroup cpp_raii_api
 *
 * This function must be called from within a catch block. It rethrows the
 * current exception and maps it to an appropriate Status code.
 *
 * @return Status::ERR_ALLOC for std::bad_alloc, Status::ERR_INTERNAL otherwise.
 */
[[nodiscard]] inline Status mapExceptionToStatus() noexcept {
  try {
    throw;
  } catch (const std::bad_alloc&) {
    return Status::ERR_ALLOC;
  } catch (...) {
    return Status::ERR_INTERNAL;
  }
}

/**
 * @brief Convert a Status code to a human-readable string.
 * @ingroup cpp_raii_api
 * @param s The Status code.
 * @return A null-terminated string representing the status.
 */
[[nodiscard]] constexpr const char* statusToString(Status s) noexcept {
  switch (s) {
  case Status::SUCCESS:
    return "SUCCESS";
  case Status::ERR_NULLPTR:
    return "NULLPTR";
  case Status::ERR_INVALID_COMM:
    return "INVALID_COMM";
  case Status::ERR_NO_BUFFER:
    return "NO_BUFFER";
  case Status::ERR_PLAN_NOT_INIT:
    return "PLAN_NOT_INIT";
  case Status::ERR_INVALID_DIM:
    return "INVALID_DIM";
  case Status::ERR_DIM_MISMATCH:
    return "DIM_MISMATCH";
  case Status::ERR_INVALID_DECOMP:
    return "INVALID_DECOMP";
  case Status::ERR_INVALID_FFTTYPE:
    return "INVALID_FFTTYPE";
  case Status::ERR_ALLOC:
    return "ALLOC";
  case Status::ERR_BACKEND:
    return "BACKEND";
  case Status::ERR_MPI:
    return "MPI";
  case Status::ERR_INVALID_LAYOUT:
    return "INVALID_LAYOUT";
  case Status::ERR_NOT_IMPL:
    return "NOT_IMPL";
  case Status::ERR_INVALID_STATE:
    return "INVALID_STATE";
  case Status::ERR_INTERNAL:
    return "INTERNAL";
  case Status::ERR_SIZE_OVERFLOW:
    return "SIZE_OVERFLOW";
  }
  // No default: lets -Wswitch-enum warn if you add a new Status and forget to map it.
  return "UNKNOWN";
}

/**
 * @brief Convert an integer status code to a human-readable string.
 * @ingroup cpp_raii_api
 * @param code The integer status code.
 * @return A null-terminated string representing the status.
 */
[[nodiscard]] constexpr const char* statusToString(int code) noexcept {
  return statusToString(static_cast<Status>(code));
}

} // namespace shafft::err

#endif // SHAFFT_ERROR_HPP
