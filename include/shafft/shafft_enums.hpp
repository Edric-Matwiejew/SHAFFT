/**
 *  @brief Enum definitions for SHAFFT C++ API.
 *  @ingroup cpp_raii_api
 */

#ifndef SHAFFT_ENUMS_HPP
#define SHAFFT_ENUMS_HPP

namespace shafft {

/**
 * @brief FFT transform direction (FORWARD / BACKWARD).
 *
 * Specifies whether to perform a forward or backward (inverse) FFT.
 *
 * @ingroup cpp_raii_api
 */
enum class FFTDirection {
  FORWARD, ///< Forward transform (time to frequency domain).
  BACKWARD ///< Backward/inverse transform (frequency to time domain).
};

/**
 * @brief FFT element/precision type (C2C / Z2Z).
 *
 * Specifies the complex data precision for the transform.
 *
 * @ingroup cpp_raii_api
 */
enum class FFTType {
  C2C, ///< Single-precision complex-to-complex (float).
  Z2Z  ///< Double-precision complex-to-complex (double).
};

/**
 * @brief Tensor layout identifier (CURRENT / INITIAL / REDISTRIBUTED).
 *
 * Identifies the data layout at different stages of the FFT lifecycle.
 *
 * @ingroup cpp_raii_api
 */
enum class TensorLayout {
  CURRENT,      ///< Current layout after most recent operation.
  INITIAL,      ///< Layout before any transforms.
  REDISTRIBUTED ///< User-visible post-forward redistributed layout.
};

} // namespace shafft

#endif // SHAFFT_ENUMS_HPP
