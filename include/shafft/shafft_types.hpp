/**
 *  @brief Core types and execution helpers for SHAFFT.
 *  @ingroup cpp_raii_api
 */

#ifndef SHAFFT_TYPES_H
#define SHAFFT_TYPES_H

#include <shafft/shafft_config.h>
#include <shafft/shafft_enums.hpp>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <mpi.h>
#include <vector>

namespace shafft {

/**
 * @brief FFT backend identifier (HIPFFT or FFTW).
 *
 * Indicates which FFT library is used at compile time.
 *
 * @ingroup cpp_raii_api
 */
enum class Backend {
  HIPFFT, ///< AMD hipFFT backend for GPU execution.
  FFTW    ///< FFTW backend for CPU execution.
};

/**
 * @brief Get the compile-time backend.
 *
 * Returns the FFT backend selected at compile time.
 *
 * @return Backend::HIPFFT or Backend::FFTW.
 * @ingroup cpp_raii_api
 */
constexpr Backend backend() noexcept {
#if SHAFFT_BACKEND_HIPFFT
  return Backend::HIPFFT;
#else
  return Backend::FFTW;
#endif
}

/**
 * @brief Get the backend name as a string.
 *
 * Returns a human-readable name for the compile-time backend.
 *
 * @return "hipFFT" or "FFTW".
 * @ingroup cpp_raii_api
 */
constexpr const char* backendName() noexcept {
#if SHAFFT_BACKEND_HIPFFT
  return "hipFFT";
#else
  return "FFTW";
#endif
}

/**
 * @brief Single-precision complex type (std::complex<float>).
 *
 * Alias for std::complex<float>. Used with FFTType::C2C transforms.
 *
 * @ingroup cpp_raii_api
 */
using complexf = std::complex<float>;

/**
 * @brief Double-precision complex type (std::complex<double>).
 *
 * Alias for std::complex<double>. Used with FFTType::Z2Z transforms.
 *
 * @ingroup cpp_raii_api
 */
using complexd = std::complex<double>;

/// @brief Plan lifecycle states.
/// @ingroup cpp_raii_api
enum class PlanState {
  UNINITIALIZED, ///< Default-constructed, no metadata.
  CONFIGURED,    ///< init() succeeded, allocSize() available.
  PLANNED        ///< plan() succeeded, ready for execute().
};

/// @brief Status and error codes.
/// @ingroup cpp_raii_api
enum class Status : int {
  SUCCESS = 0,             ///< Operation succeeded.
  ERR_NULLPTR = 1,         ///< A required pointer argument was null.
  ERR_INVALID_COMM = 2,    ///< Invalid or unsupported MPI communicator.
  ERR_NO_BUFFER = 3,       ///< Required data/work buffer was not set.
  ERR_PLAN_NOT_INIT = 4,   ///< Plan or subplan not initialized.
  ERR_INVALID_DIM = 5,     ///< Invalid dimension/rank/size.
  ERR_DIM_MISMATCH = 6,    ///< Dimension mismatch between inputs.
  ERR_INVALID_DECOMP = 7,  ///< Invalid or unsupported slab decomposition.
  ERR_INVALID_FFTTYPE = 8, ///< Unsupported FFTType.
  ERR_ALLOC = 9,           ///< Memory allocation failure.
  ERR_BACKEND = 10,        ///< Local FFT backend failure.
  ERR_MPI = 11,            ///< MPI failure.
  ERR_INVALID_LAYOUT = 12, ///< Layout parameters don't match expected distribution.
  ERR_SIZE_OVERFLOW = 13,  ///< Size exceeds INT_MAX.
  ERR_NOT_IMPL = 14,       ///< Feature not yet implemented.
  ERR_INVALID_STATE = 15,  ///< Operation not valid in current plan state.
  ERR_INTERNAL = 16        ///< Uncategorized internal error.
};

/// @brief Strategy for automatic decomposition selection.
/// @ingroup cpp_raii_api
enum class DecompositionStrategy {
  MAXIMIZE_NDA, ///< Maximize distributed axes
  MINIMIZE_NDA  ///< Minimize distributed axes
};

/// @brief Plan type discriminator.
/// @ingroup cpp_raii_api
enum class PlanType {
  PLAN_ND, ///< N-dimensional distributed FFT plan
  PLAN_1D  ///< 1-dimensional distributed FFT plan
};

/// @brief Output-layout policy for forward FFT transforms.
/// @ingroup cpp_raii_api
enum class TransformLayout {
  REDISTRIBUTED, ///< Keep post-forward redistributed layout (no final redistribution).
  INITIAL        ///< Restore initial layout after forward transform.
};

/// @brief Configuration resolve policy.
/// @ingroup cpp_raii_api
enum class ConfigPolicy {
  AUTO_ADJUST = 0, ///< Adjust hints using fallback logic (default; zero-init safe).
  EXACT = 1        ///< Fail if hints cannot be satisfied exactly.
};

/**
 * @brief Product of the first @p ndim elements.
 * @ingroup cpp_raii_api
 * @tparam T Input element type.
 * @tparam U Accumulator/return type.
 * @param array Pointer to at least @p ndim elements.
 * @param ndim Number of elements to multiply.
 * @return Product cast to U.
 */
template <typename T, typename U>
U product(T* array, int ndim) {
  U prod = static_cast<U>(1);
  for (int i = 0; i < ndim; i++) {
    prod *= static_cast<U>(array[i]);
  }
  return prod;
}

} // namespace shafft

#endif // SHAFFT_TYPES_H
