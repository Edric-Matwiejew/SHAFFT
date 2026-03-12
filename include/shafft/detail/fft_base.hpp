/** @brief Abstract base class for FFT Types.
 *  @ingroup cpp_raii_api
 */

#ifndef SHAFFT_FFT_BASE_HPP
#define SHAFFT_FFT_BASE_HPP

#include <shafft/shafft_config.h>
#include <shafft/shafft_types.hpp>

#include <cstddef>

#if SHAFFT_BACKEND_HIPFFT
#include <hip/hip_runtime.h>
#endif

namespace shafft {

/**
 * @brief Abstract base class for FFT plans (FFTND, FFT1D).
 * @ingroup cpp_raii_api
 */
class FFT {
public:
  /// @cond INTERNAL
  virtual ~FFT() noexcept = default;
  /// @endcond

  /**
   * @brief Release internal resources.
   *
   * Call before MPI_Finalize() if plan outlives MPI.
   */
  virtual void release() noexcept = 0;

  /**
   * @brief Attach data and work buffers.
   * @param data Data buffer (at least allocSize() elements).
   * @param work Work buffer (at least allocSize() elements).
   * @return 0 on success.
   */
  [[nodiscard]] virtual int setBuffersRaw(void* data, void* work) noexcept = 0;

  /**
   * @brief Retrieve current buffer pointers.
   *
   * Buffers may be swapped after execute().
   *
   * @param[out] data Current data buffer.
   * @param[out] work Current work buffer.
   * @return 0 on success.
   */
  [[nodiscard]] virtual int getBuffersRaw(void** data, void** work) noexcept = 0;

  /**
   * @brief Create backend FFT plans.
   *
   * Must be called after init() succeeds. For GPU backends, buffers must be
   * attached before calling plan(). For FFTW, buffers may be set later.
   * Calling plan() more than once returns Status::ERR_INVALID_STATE.
   *
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] virtual int plan() noexcept = 0;

  /**
   * @brief Execute the FFT.
   * @param direction FORWARD or BACKWARD.
   * @return Status code (0 on success).
   */
  [[nodiscard]] virtual int execute(FFTDirection direction) noexcept = 0;

  /**
   * @brief Apply normalization to the transformed data.
   * @return Status code (0 on success).
   */
  [[nodiscard]] virtual int normalize() noexcept = 0;

  /**
   * @brief Get the required buffer allocation size (in elements).
   * @return Number of elements needed for data/work buffers.
   */
  [[nodiscard]] virtual size_t allocSize() const noexcept = 0;

  /**
   * @brief Get the total number of elements in the global tensor.
   * @return Product of all global dimensions.
   */
  [[nodiscard]] virtual size_t globalSize() const noexcept = 0;

  /**
   * @brief Get the number of dimensions.
   * @return Number of tensor dimensions (1 for FFT1D, N for FFTND).
   */
  [[nodiscard]] virtual int ndim() const noexcept = 0;

  /// @brief Check if the plan is valid (alias for isConfigured).
  explicit operator bool() const noexcept { return isConfigured(); }

  /**
   * @brief Check if this rank participates in the FFT.
   *
   * Inactive ranks can safely call execute()/normalize() (no-ops).
   */
  [[nodiscard]] virtual bool isActive() const noexcept = 0;

  /**
   * @brief Get the FFT type (C2C or Z2Z).
   * @return The FFTType of this plan.
   */
  [[nodiscard]] virtual FFTType fftType() const noexcept = 0;

  /**
   * @brief Get the current plan state.
   * @return The PlanState of this plan.
   */
  [[nodiscard]] PlanState state() const noexcept { return state_; }

  /**
   * @brief Check if the plan is at least configured (init() succeeded).
   * @return true if state >= CONFIGURED.
   */
  [[nodiscard]] bool isConfigured() const noexcept { return state_ >= PlanState::CONFIGURED; }

  /**
   * @brief Check if the plan is fully planned and ready for execution.
   * @return true if state == PLANNED.
   */
  [[nodiscard]] bool isPlanned() const noexcept { return state_ == PlanState::PLANNED; }

#if SHAFFT_BACKEND_HIPFFT
  /**
   * @brief Set the HIP stream for GPU execution.
   * @param stream HIP stream to use.
   * @return Status code (0 on success).
   */
  [[nodiscard]] virtual int setStream(hipStream_t stream) noexcept = 0;
#endif

  // Non-copyable
  FFT(const FFT&) = delete;
  FFT& operator=(const FFT&) = delete;

protected:
  /// @cond INTERNAL
  /// @brief Plan lifecycle state.
  PlanState state_ = PlanState::UNINITIALIZED;

  // Protected constructor - cannot instantiate FFT directly
  FFT() noexcept = default;

  // Movable via derived classes
  FFT(FFT&&) noexcept = default;
  FFT& operator=(FFT&&) noexcept = default;
  /// @endcond
};

} // namespace shafft

#endif // SHAFFT_FFT_BASE_HPP
