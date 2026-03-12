/**
 *  @brief C++ interface for SHAFFT.
 *  @ingroup cpp_raii_api
 */

#ifndef SHAFFT_CPP_H
#define SHAFFT_CPP_H

#include <shafft/detail/fft_base.hpp>
#include <shafft/shafft.h> // C config structs
#include <shafft/shafft_config.h>
#include <shafft/shafft_types.hpp>

#include <cstddef>
#include <mpi.h>
#include <vector>

#if SHAFFT_BACKEND_HIPFFT
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#endif
#if SHAFFT_BACKEND_FFTW
#include <fftw3.h>
#endif

/// @cond INTERNAL
// Opaque declarations for PIMPL
namespace shafft::detail {
class PlanBase;
struct FFT1DPlan;
struct FFTNDPlan;
} // namespace shafft::detail
/// @endcond

namespace shafft {

/**
 * @brief N-dimensional distributed FFT plan with RAII semantics.
 * @ingroup cpp_raii_api
 *
 * Manages plan lifetime automatically. User owns data and work buffers.
 *
 * @note Typical usage flow:
 * 1. Call configurationND() to compute decomposition parameters
 * 2. Construct FFTND and call init() with the computed parameters
 * 3. Allocate buffers using allocSize() and call setBuffers()
 * 4. Call plan() to create backend FFT plans
 * 5. Call execute() for forward/backward transforms
 * 6. Call normalize() after a forward-backward pair to restore scale
 */
class FFTND : public FFT {
public:
  /// @brief Default constructor. Creates an uninitialized plan.
  FFTND() noexcept = default;

  /// @brief Destructor. Releases all internal resources.
  ~FFTND() noexcept override;

  /**
   * @brief Release all internal resources.
   *
   * Call before MPI_Finalize() if the plan outlives MPI.
   * After release(), the plan is uninitialized.
   */
  void release() noexcept override;

  /**
   * @brief Move constructor.
   *
   * Transfers ownership of backend resources from @p other.
   * The moved-from plan becomes uninitialized.
   *
   * @param other Plan to move from.
   */
  FFTND(FFTND&& other) noexcept;

  /**
   * @brief Move assignment operator.
   *
   * Releases current resources and takes ownership from @p other.
   * The moved-from plan becomes uninitialized.
   *
   * @param other Plan to move from.
   * @return Reference to this plan.
   */
  FFTND& operator=(FFTND&& other) noexcept;

  FFTND(const FFTND&) = delete;
  FFTND& operator=(const FFTND&) = delete;

  /**
   * @brief Initialize plan with Cartesian process grid.
   *
   * Call configurationND() first to compute @p commDims.
   * After init(), call setBuffers() then plan() before execute().
   *
   * @param commDims   Process grid dimensions.
   * @param dimensions Global tensor dimensions.
   * @param type       FFT precision (C2C or Z2Z).
   * @param comm       MPI communicator.
   * @param output     Forward output-layout policy.
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int init(const std::vector<int>& commDims,
                         const std::vector<size_t>& dimensions,
                         FFTType type,
                         MPI_Comm comm,
                         TransformLayout output = TransformLayout::REDISTRIBUTED) noexcept;

  /**
   * @brief Create backend FFT plans.
   *
   * Must be called after init().
   * For the FFTW backend, dummy buffers are allocated and used for planning
   * if they are not already set.
   *
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int plan() noexcept override;

  /**
   * @brief Attach data and work buffers.
   * @param data Data buffer (allocSize() elements).
   * @param work Work buffer (allocSize() elements).
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int setBuffers(complexf* data, complexf* work) noexcept;
  [[nodiscard]] int setBuffers(complexd* data, complexd* work) noexcept;

  /**
   * @brief Retrieve current buffer pointers.
   *
   * Buffers may be swapped after execute(); always call this to locate output.
   *
   * @param[out] data Current data buffer.
   * @param[out] work Current work buffer.
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int getBuffers(complexf** data, complexf** work) noexcept;
  [[nodiscard]] int getBuffers(complexd** data, complexd** work) noexcept;

#if SHAFFT_BACKEND_HIPFFT
  /**
   * @brief Attach HIP buffers (single precision).
   * @param data Device data buffer (allocSize() elements).
   * @param work Device work buffer (allocSize() elements).
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int setBuffers(hipFloatComplex* data, hipFloatComplex* work) noexcept;

  /**
   * @brief Attach HIP buffers (double precision).
   * @param data Device data buffer (allocSize() elements).
   * @param work Device work buffer (allocSize() elements).
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int setBuffers(hipDoubleComplex* data, hipDoubleComplex* work) noexcept;

  /**
   * @brief Retrieve current HIP buffers (single precision).
   * @param[out] data Current device data buffer.
   * @param[out] work Current device work buffer.
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int getBuffers(hipFloatComplex** data, hipFloatComplex** work) noexcept;

  /**
   * @brief Retrieve current HIP buffers (double precision).
   * @param[out] data Current device data buffer.
   * @param[out] work Current device work buffer.
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int getBuffers(hipDoubleComplex** data, hipDoubleComplex** work) noexcept;
#endif
#if SHAFFT_BACKEND_FFTW
  /**
   * @brief Attach FFTW buffers (double precision).
   * @param data Host data buffer (allocSize() elements).
   * @param work Host work buffer (allocSize() elements).
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int setBuffers(fftw_complex* data, fftw_complex* work) noexcept;

  /**
   * @brief Attach FFTW buffers (single precision).
   * @param data Host data buffer (allocSize() elements).
   * @param work Host work buffer (allocSize() elements).
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int setBuffers(fftwf_complex* data, fftwf_complex* work) noexcept;

  /**
   * @brief Retrieve current FFTW buffers (double precision).
   * @param[out] data Current host data buffer.
   * @param[out] work Current host work buffer.
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int getBuffers(fftw_complex** data, fftw_complex** work) noexcept;

  /**
   * @brief Retrieve current FFTW buffers (single precision).
   * @param[out] data Current host data buffer.
   * @param[out] work Current host work buffer.
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int getBuffers(fftwf_complex** data, fftwf_complex** work) noexcept;
#endif

  /**
   * @brief Execute the FFT.
   * @param direction FORWARD or BACKWARD.
   * @return Status code (0 on success).
   */
  [[nodiscard]] int execute(FFTDirection direction) noexcept override;

  /**
   * @brief Apply symmetric normalization (1/sqrt(N) per transform).
   *
   * After a forward-backward pair, calling normalize() restores original scale.
   *
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int normalize() noexcept override;

  /**
   * @brief Get required buffer size in complex elements.
   */
  [[nodiscard]] size_t allocSize() const noexcept override;

  /**
   * @brief Get total global tensor size (product of dimensions).
   */
  [[nodiscard]] size_t globalSize() const noexcept override;

  /**
   * @brief Get number of dimensions.
   */
  [[nodiscard]] int ndim() const noexcept override;

  /**
   * @brief Get FFT precision (C2C or Z2Z).
   */
  [[nodiscard]] FFTType fftType() const noexcept override;

  /**
   * @brief Query local tensor layout.
   * @param[out] subsize Local extent per axis.
   * @param[out] offset  Global offset per axis.
   * @param layout       CURRENT, INITIAL, or REDISTRIBUTED.
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int getLayout(std::vector<size_t>& subsize,
                              std::vector<size_t>& offset,
                              TensorLayout layout = TensorLayout::CURRENT) const noexcept;

  /**
   * @brief Query axis distribution.
   * @param[out] ca Contiguous (non-distributed) axes.
   * @param[out] da Distributed axes.
   * @param layout  CURRENT, INITIAL, or REDISTRIBUTED.
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int getAxes(std::vector<int>& ca,
                            std::vector<int>& da,
                            TensorLayout layout = TensorLayout::CURRENT) const noexcept;

  /**
   * @brief Check if this rank participates in computation.
   *
   * Inactive ranks have no local data; execute() and normalize() are no-ops.
   */
  [[nodiscard]] bool isActive() const noexcept override;

  /**
   * @brief Attach buffers (type-erased).
   * @param data Data buffer.
   * @param work Work buffer.
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int setBuffersRaw(void* data, void* work) noexcept override;

  /**
   * @brief Retrieve current buffer pointers (type-erased).
   * @param[out] data Current data buffer.
   * @param[out] work Current work buffer.
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int getBuffersRaw(void** data, void** work) noexcept override;

#if SHAFFT_BACKEND_HIPFFT
  /**
   * @brief Set HIP stream for subsequent operations.
   * @param stream HIP stream handle to use for execution.
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int setStream(hipStream_t stream) noexcept override;
#endif

  /**
   * @brief Get a duplicated communicator from this plan.
   *
   * Returns MPI_COMM_NULL for inactive ranks. The caller must
   * call MPI_Comm_free() on the returned communicator when done.
   * Valid only after plan() succeeds.
   *
   * @param[out] outComm Receives duplicated communicator.
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int getCommunicator(MPI_Comm* outComm) const noexcept;

  /**
   * @brief Initialize plan from a resolved config object.
   *
   * Auto-resolves if SHAFFT_CONFIG_RESOLVED is not set.
   * Communicator is read from the config struct (worldComm).
   *
   * @param cfg    Config struct (resolved or will be auto-resolved).
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int init(shafft_nd_config_t& cfg) noexcept;

private:
  detail::FFTNDPlan* data_ = nullptr;
}; // class FFTND

/**
 * @brief 1-dimensional distributed FFT plan with RAII semantics.
 * @ingroup cpp_raii_api
 *
 * Uses block distribution: rank r owns [localStart, localStart + localN).
 * allocSize() may exceed localN due to padding.
 *
 * @note Typical usage flow:
 * 1. Call configuration1D() to compute local layout
 * 2. Construct FFT1D and call init() with the computed parameters
 * 3. Allocate buffers using allocSize() and call setBuffers()
 * 4. Call plan() to create backend FFT plans
 * 5. Call execute() for forward/backward transforms
 * 6. Call normalize() after a forward-backward pair to restore scale
 */
class FFT1D : public FFT {
public:
  /// @brief Default constructor (uninitialized).
  FFT1D() noexcept = default;

  /// @brief Destructor.
  ~FFT1D() noexcept override;

  /**
   * @brief Release all internal resources.
   *
   * Call before MPI_Finalize() if the plan outlives MPI.
   */
  void release() noexcept override;

  /**
   * @brief Move constructor.
   *
   * Transfers ownership of backend resources from @p other.
   * The moved-from plan becomes uninitialized.
   *
   * @param other Plan to move from.
   */
  FFT1D(FFT1D&& other) noexcept;

  /**
   * @brief Move assignment operator.
   *
   * Releases current resources and takes ownership from @p other.
   * The moved-from plan becomes uninitialized.
   *
   * @param other Plan to move from.
   * @return Reference to this plan.
   */
  FFT1D& operator=(FFT1D&& other) noexcept;

  // Non-copyable
  FFT1D(const FFT1D&) = delete;
  FFT1D& operator=(const FFT1D&) = delete;

  /**
   * @brief Initialize plan.
   *
   * Call configuration1D() first to compute layout parameters.
   *
   * @param globalN    Global FFT size.
   * @param localN     Local element count for this rank.
   * @param localStart This rank's offset in global array.
   * @param precision  FFT precision (C2C or Z2Z).
   * @param comm       MPI communicator.
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int
  init(size_t globalN, size_t localN, size_t localStart, FFTType precision, MPI_Comm comm) noexcept;

  /**
   * @brief Create backend FFT plans.
   *
   * Must be called after init().
   * For FFTW, buffers may be set later; dummy buffers are used for planning.
   * Calling plan() more than once is an error.
   *
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int plan() noexcept override;

  /**
   * @brief Attach data and work buffers.
   *
   * Buffers must be allocated and have at least allocSize() elements.
   *
   * @param data Input buffer (allocSize() elements).
   * @param work Output buffer (allocSize() elements).
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int setBuffers(complexf* data, complexf* work) noexcept;
  [[nodiscard]] int setBuffers(complexd* data, complexd* work) noexcept;

  /**
   * @brief Retrieve current buffer pointers.
   * @param[out] data Current data buffer.
   * @param[out] work Current work buffer.
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int getBuffers(complexf** data, complexf** work) noexcept;

  /**
   * @brief Retrieve current buffer pointers (double precision).
   * @param[out] data Current data buffer.
   * @param[out] work Current work buffer.
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int getBuffers(complexd** data, complexd** work) noexcept;

#if SHAFFT_BACKEND_HIPFFT
  /**
   * @brief Attach HIP buffers (single precision).
   * @param data Device data buffer (allocSize() elements).
   * @param work Device work buffer (allocSize() elements).
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int setBuffers(hipFloatComplex* data, hipFloatComplex* work) noexcept;

  /**
   * @brief Attach HIP buffers (double precision).
   * @param data Device data buffer (allocSize() elements).
   * @param work Device work buffer (allocSize() elements).
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int setBuffers(hipDoubleComplex* data, hipDoubleComplex* work) noexcept;

  /**
   * @brief Retrieve current HIP buffers (single precision).
   * @param[out] data Current device data buffer.
   * @param[out] work Current device work buffer.
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int getBuffers(hipFloatComplex** data, hipFloatComplex** work) noexcept;

  /**
   * @brief Retrieve current HIP buffers (double precision).
   * @param[out] data Current device data buffer.
   * @param[out] work Current device work buffer.
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int getBuffers(hipDoubleComplex** data, hipDoubleComplex** work) noexcept;
#endif
#if SHAFFT_BACKEND_FFTW
  /**
   * @brief Attach FFTW buffers (double precision).
   * @param data Host data buffer (allocSize() elements).
   * @param work Host work buffer (allocSize() elements).
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int setBuffers(fftw_complex* data, fftw_complex* work) noexcept;

  /**
   * @brief Attach FFTW buffers (single precision).
   * @param data Host data buffer (allocSize() elements).
   * @param work Host work buffer (allocSize() elements).
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int setBuffers(fftwf_complex* data, fftwf_complex* work) noexcept;

  /**
   * @brief Retrieve current FFTW buffers (double precision).
   * @param[out] data Current host data buffer.
   * @param[out] work Current host work buffer.
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int getBuffers(fftw_complex** data, fftw_complex** work) noexcept;

  /**
   * @brief Retrieve current FFTW buffers (single precision).
   * @param[out] data Current host data buffer.
   * @param[out] work Current host work buffer.
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int getBuffers(fftwf_complex** data, fftwf_complex** work) noexcept;
#endif

  /**
   * @brief Execute the transform.
   * @param direction FORWARD or BACKWARD.
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int execute(FFTDirection direction) noexcept override;

  /**
   * @brief Apply symmetric normalization (1/sqrt(N) per transform).
   *
   * After a forward-backward pair, calling normalize() restores original scale.
   *
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int normalize() noexcept override;

  /**
   * @brief Get global shape as length-1 vector.
   */
  [[nodiscard]] std::vector<size_t> globalShape() const noexcept;

  /**
   * @brief Get global FFT size.
   */
  [[nodiscard]] size_t globalSize() const noexcept override;

  /**
   * @brief Get number of dimensions (always 1).
   */
  [[nodiscard]] int ndim() const noexcept override;

  /**
   * @brief Get FFT precision (C2C or Z2Z).
   */
  [[nodiscard]] FFTType fftType() const noexcept override;

  /**
   * @brief Get local element count (before padding).
   */
  [[nodiscard]] size_t localSize() const noexcept;

  /**
   * @brief Get required buffer size in complex elements.
   *
   * May exceed localSize() due to padding.
   */
  [[nodiscard]] size_t allocSize() const noexcept override;

  /**
   * @brief Query local layout.
   * @param[out] localShape Local size as length-1 vector.
   * @param[out] offset     Local offset as length-1 vector.
   * @param layout          CURRENT, INITIAL, or REDISTRIBUTED.
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int getLayout(std::vector<size_t>& localShape,
                              std::vector<size_t>& offset,
                              TensorLayout layout = TensorLayout::CURRENT) const noexcept;

  /**
   * @brief Query axis distribution.
   *
   * Returns ca={} and da={0} (the single axis is distributed).
   *
   * @param[out] ca Contiguous axes (empty).
   * @param[out] da Distributed axes ({0}).
   * @param layout  CURRENT, INITIAL, or REDISTRIBUTED.
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int getAxes(std::vector<int>& ca,
                            std::vector<int>& da,
                            TensorLayout layout = TensorLayout::CURRENT) const noexcept;

  /// @brief Check if this rank participates in computation.
  [[nodiscard]] bool isActive() const noexcept override;

  /**
   * @brief Attach buffers (type-erased).
   * @param data Data buffer.
   * @param work Work buffer.
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int setBuffersRaw(void* data, void* work) noexcept override;

  /**
   * @brief Retrieve current buffer pointers (type-erased).
   * @param[out] data Current data buffer.
   * @param[out] work Current work buffer.
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int getBuffersRaw(void** data, void** work) noexcept override;

#if SHAFFT_BACKEND_HIPFFT
  /**
   * @brief Set HIP stream for subsequent operations.
   * @param stream HIP stream handle to use for execution.
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int setStream(hipStream_t stream) noexcept override;
#endif

  /**
   * @brief Get a duplicated communicator from this plan.
   *
   * Returns MPI_COMM_NULL for inactive ranks. The caller must
   * call MPI_Comm_free() on the returned communicator when done.
   *
   * @param[out] outComm Receives duplicated communicator.
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int getCommunicator(MPI_Comm* outComm) const noexcept;

  /**
   * @brief Initialize plan from a resolved 1-D config object.
   *
   * Auto-resolves if SHAFFT_CONFIG_RESOLVED is not set.
   * Communicator is read from the config struct (worldComm).
   *
   * @param cfg  Config struct (resolved or will be auto-resolved).
   * @return 0 on success, non-zero on error.
   */
  [[nodiscard]] int init(shafft_1d_config_t& cfg) noexcept;

private:
  detail::FFT1DPlan* data_ = nullptr;
}; // class FFT1D

#if SHAFFT_BACKEND_HIPFFT
/**
 * @brief Set HIP stream for GPU operations.
 * @ingroup cpp_lowlevel_api
 * @param plan   Plan handle.
 * @param stream HIP stream.
 * @return 0 on success, non-zero on error.
 */
int setStream(detail::PlanBase* plan, hipStream_t stream);

/**
 * @brief Retrieve HIP buffers (single precision).
 * @ingroup cpp_lowlevel_api
 * @param plan       Plan handle.
 * @param[out] data  Current device data buffer.
 * @param[out] work  Current device work buffer.
 * @return 0 on success, non-zero on error.
 */
int getBuffers(detail::PlanBase* plan, hipFloatComplex** data, hipFloatComplex** work) noexcept;

/**
 * @brief Retrieve HIP buffers (double precision).
 * @ingroup cpp_lowlevel_api
 * @param plan       Plan handle.
 * @param[out] data  Current device data buffer.
 * @param[out] work  Current device work buffer.
 * @return 0 on success, non-zero on error.
 */
int getBuffers(detail::PlanBase* plan, hipDoubleComplex** data, hipDoubleComplex** work) noexcept;

/**
 * @brief Set HIP buffers (single precision).
 * @ingroup cpp_lowlevel_api
 * @param plan  Plan handle.
 * @param data  Device data buffer (allocSize() elements).
 * @param work  Device work buffer (allocSize() elements).
 * @return 0 on success, non-zero on error.
 */
int setBuffers(detail::PlanBase* plan, hipFloatComplex* data, hipFloatComplex* work) noexcept;

/**
 * @brief Set HIP buffers (double precision).
 * @ingroup cpp_lowlevel_api
 * @param plan  Plan handle.
 * @param data  Device data buffer (allocSize() elements).
 * @param work  Device work buffer (allocSize() elements).
 * @return 0 on success, non-zero on error.
 */
int setBuffers(detail::PlanBase* plan, hipDoubleComplex* data, hipDoubleComplex* work) noexcept;
#endif
#if SHAFFT_BACKEND_FFTW
/**
 * @brief Retrieve FFTW buffers (double precision).
 * @ingroup cpp_lowlevel_api
 * @param plan       Plan handle.
 * @param[out] data  Current host data buffer.
 * @param[out] work  Current host work buffer.
 * @return 0 on success, non-zero on error.
 */
int getBuffers(detail::PlanBase* plan, fftw_complex** data, fftw_complex** work) noexcept;

/**
 * @brief Retrieve FFTW buffers (single precision).
 * @ingroup cpp_lowlevel_api
 * @param plan       Plan handle.
 * @param[out] data  Current host data buffer.
 * @param[out] work  Current host work buffer.
 * @return 0 on success, non-zero on error.
 */
int getBuffers(detail::PlanBase* plan, fftwf_complex** data, fftwf_complex** work) noexcept;

/**
 * @brief Set FFTW buffers (double precision).
 * @ingroup cpp_lowlevel_api
 * @param plan  Plan handle.
 * @param data  Host data buffer (allocSize() elements).
 * @param work  Host work buffer (allocSize() elements).
 * @return 0 on success, non-zero on error.
 */
int setBuffers(detail::PlanBase* plan, fftw_complex* data, fftw_complex* work) noexcept;

/**
 * @brief Set FFTW buffers (single precision).
 * @ingroup cpp_lowlevel_api
 * @param plan  Plan handle.
 * @param data  Host data buffer (allocSize() elements).
 * @param work  Host work buffer (allocSize() elements).
 * @return 0 on success, non-zero on error.
 */
int setBuffers(detail::PlanBase* plan, fftwf_complex* data, fftwf_complex* work) noexcept;
#endif

/**
 * @brief Attach buffers using portable complex types.
 * @ingroup cpp_lowlevel_api
 *
 * GPU backends require device memory; CPU backends require host memory.
 *
 * @param plan Plan handle.
 * @param data Data buffer.
 * @param work Work buffer.
 * @return 0 on success, non-zero on error.
 */
int setBuffers(detail::PlanBase* plan, complexf* data, complexf* work) noexcept;
int setBuffers(detail::PlanBase* plan, complexd* data, complexd* work) noexcept;

/**
 * @brief Retrieve current buffer pointers.
 * @ingroup cpp_lowlevel_api
 *
 * Buffers may be swapped after execute().
 *
 * @param plan        Plan handle.
 * @param[out] data   Current data buffer.
 * @param[out] work   Current work buffer.
 * @return 0 on success, non-zero on error.
 */
int getBuffers(detail::PlanBase* plan, complexf** data, complexf** work) noexcept;
int getBuffers(detail::PlanBase* plan, complexd** data, complexd** work) noexcept;

/**
 * @brief Allocate buffer for the current backend.
 * @ingroup cpp_raii_api
 * @ingroup cpp_lowlevel_api
 *
 * Uses hipMalloc on GPU, standard allocation on CPU.
 *
 * @param count       Number of complex elements.
 * @param[out] buf    Allocated buffer.
 * @return 0 on success, non-zero on error.
 */
int allocBuffer(size_t count, complexf** buf) noexcept;
int allocBuffer(size_t count, complexd** buf) noexcept;

/**
 * @brief Free buffer allocated with allocBuffer().
 * @ingroup cpp_raii_api
 * @ingroup cpp_lowlevel_api
 * @param buf Buffer to free (nullptr safe).
 * @return 0 on success, non-zero on error.
 */
int freeBuffer(complexf* buf) noexcept;
int freeBuffer(complexd* buf) noexcept;

/**
 * @brief Copy from host to SHAFFT buffer.
 * @ingroup cpp_raii_api
 * @ingroup cpp_lowlevel_api
 *
 * Performs copy to a backend buffer.
 *
 * @param dst   Destination buffer.
 * @param src   Source host memory.
 * @param count Number of complex elements.
 * @return 0 on success, non-zero on error.
 */
int copyToBuffer(complexf* dst, const complexf* src, size_t count) noexcept;
int copyToBuffer(complexd* dst, const complexd* src, size_t count) noexcept;

/**
 * @brief Copy from SHAFFT buffer to host.
 * @ingroup cpp_raii_api
 * @ingroup cpp_lowlevel_api
 *
 * Performs copy from a backend buffer to host memory.
 *
 * @param dst   Destination host memory.
 * @param src   Source buffer.
 * @param count Number of complex elements.
 * @return 0 on success, non-zero on error.
 */
int copyFromBuffer(complexf* dst, const complexf* src, size_t count) noexcept;
int copyFromBuffer(complexd* dst, const complexd* src, size_t count) noexcept;

/**
 * @brief Allocate N-D plan object (C-style API).
 * @ingroup cpp_lowlevel_api
 *
 * Prefer the RAII FFTND class. Release with destroy().
 *
 * @param[out] out Plan pointer.
 * @return 0 on success, non-zero on error.
 */
int planNDCreate(detail::FFTNDPlan** out);

/**
 * @brief Allocate 1-D plan object (C-style API).
 * @ingroup cpp_lowlevel_api
 *
 * Prefer the RAII FFT1D class. Release with destroy().
 *
 * @param[out] out Plan pointer.
 * @return 0 on success, non-zero on error.
 */
int FFT1DCreate(detail::FFT1DPlan** out);

/**
 * @brief Compute process grid and local layout for N-D distributed FFT.
 * @ingroup cpp_raii_api
 *
 * Fallback order: commDims (if fully specified) -> nda -> strategy.
 *
 * @param size        Global tensor dimensions.
 * @param precision   FFT precision (C2C or Z2Z).
 * @param commDims    Process grid [in/out]; zeros = auto.
 * @param nda         Distributed axes [in/out]; 0 = auto.
 * @param[out] subsize Local extent per axis.
 * @param[out] offset  Global offset per axis.
 * @param[out] commSize Active rank count.
 * @param strategy    Fallback: MAXIMIZE_NDA or MINIMIZE_NDA.
 * @param memLimit    Per-rank memory limit in bytes (0 = unlimited).
 * @param comm        MPI communicator.
 * @return 0 on success, non-zero on error.
 */
int configurationND(const std::vector<size_t>& size,
                    FFTType precision,
                    std::vector<int>& commDims,
                    int& nda,
                    std::vector<size_t>& subsize,
                    std::vector<size_t>& offset,
                    int& commSize,
                    DecompositionStrategy strategy,
                    size_t memLimit,
                    MPI_Comm comm);

/**
 * @brief Compute local layout for 1D distributed FFT.
 * @ingroup cpp_raii_api
 * @ingroup cpp_lowlevel_api
 *
 * Call before FFT1D::init() to obtain layout parameters.
 *
 * @param globalN         Global FFT size.
 * @param[out] localN     Local element count.
 * @param[out] localStart Local offset in global array.
 * @param precision       FFT precision (C2C or Z2Z).
 * @param comm            MPI communicator.
 * @return 0 on success, non-zero on error.
 */
int configuration1D(
    size_t globalN, size_t& localN, size_t& localStart, FFTType precision, MPI_Comm comm);

/**
 * @brief Initialize N-D plan (C-style API).
 * @ingroup cpp_raii_api
 * @ingroup cpp_lowlevel_api
 *
 * Prefer FFTND::init() for RAII interface.
 *
 * @param plan       Plan from planNDCreate().
 * @param commDims   Process grid dimensions.
 * @param dimensions Global tensor dimensions.
 * @param precision  FFT precision (C2C or Z2Z).
 * @param comm       MPI communicator.
 * @return 0 on success, non-zero on error.
 */
int planND(detail::FFTNDPlan* plan,
           const std::vector<int>& commDims,
           const std::vector<size_t>& dimensions,
           FFTType precision,
           MPI_Comm comm);

/**
 * @brief Release plan resources and null pointer (C-style API).
 * @ingroup cpp_lowlevel_api
 *
 * Does not free user buffers. RAII classes handle this automatically.
 *
 * @param[in,out] plan Plan pointer; set to nullptr on success.
 * @return 0 on success, non-zero on error.
 */
int destroy(detail::PlanBase** plan);

/**
 * @brief Query tensor layout (C-style API).
 * @ingroup cpp_lowlevel_api
 *
 * @param plan     Plan handle.
 * @param[out] subsize Local extent per axis.
 * @param[out] offset  Global offset per axis.
 * @param layout   Layout to query (CURRENT, INITIAL, or REDISTRIBUTED).
 * @return 0 on success, non-zero on error.
 */
int getLayout(const detail::PlanBase* plan,
              std::vector<size_t>& subsize,
              std::vector<size_t>& offset,
              TensorLayout layout);

/**
 * @brief Query axis distribution (C-style API).
 * @ingroup cpp_lowlevel_api
 *
 * @param plan   Plan handle.
 * @param[out] ca Contiguous (non-distributed) axes.
 * @param[out] da Distributed axes.
 * @param layout Layout to query (CURRENT, INITIAL, or REDISTRIBUTED).
 * @return 0 on success, non-zero on error.
 */
int getAxes(const detail::PlanBase* plan,
            std::vector<int>& ca,
            std::vector<int>& da,
            TensorLayout layout);

/**
 * @brief Get required buffer size in complex elements (C-style API).
 * @ingroup cpp_lowlevel_api
 *
 * @param plan            Plan handle.
 * @param[out] localAllocSize Required buffer size in complex elements.
 * @return 0 on success, non-zero on error.
 */
int getAllocSize(const detail::PlanBase* plan, size_t& localAllocSize);

/**
 * @brief Execute transform (C-style API).
 * @ingroup cpp_lowlevel_api
 *
 * @param plan      Plan handle.
 * @param direction FORWARD or BACKWARD.
 * @return 0 on success, non-zero on error.
 */
int execute(detail::PlanBase* plan, FFTDirection direction);

/**
 * @brief Apply normalization (C-style API).
 * @ingroup cpp_lowlevel_api
 *
 * @param plan Plan handle.
 * @return 0 on success, non-zero on error.
 */
int normalize(detail::PlanBase* plan);

/**
 * @brief Get the name of the FFT backend used at compile time.
 * @ingroup cpp_raii_api
 * @ingroup cpp_lowlevel_api
 * @return "FFTW" or "hipFFT".
 */
inline const char* getBackendName() noexcept {
  return SHAFFT_BACKEND_NAME;
}

/**
 * @brief Library version information.
 * @ingroup cpp_raii_api
 * @ingroup cpp_lowlevel_api
 */
struct Version {
  /**@brief Major version number. */
  int major; ///< Major version number.
  /**@brief Minor version number. */
  int minor; ///< Minor version number.
  /**@brief Patch version number. */
  int patch; ///< Patch version number.
};

/**
 * @brief Get the library version as a struct.
 * @ingroup cpp_raii_api
 * @ingroup cpp_lowlevel_api
 *
 * @return Version with @c major, @c minor, and @c patch fields populated.
 */
inline Version getVersion() noexcept {
  return {SHAFFT_VERSION_MAJOR, SHAFFT_VERSION_MINOR, SHAFFT_VERSION_PATCH};
}

/**
 * @brief Get the library version as a string (e.g., "1.1.0-alpha").
 * @ingroup cpp_raii_api
 */
inline const char* getVersionString() noexcept {
  return SHAFFT_STRINGIFY(SHAFFT_VERSION_MAJOR) "." SHAFFT_STRINGIFY(
      SHAFFT_VERSION_MINOR) "." SHAFFT_STRINGIFY(SHAFFT_VERSION_PATCH) SHAFFT_VERSION_SUFFIX;
}

/**
 * @brief Finalize library and release backend resources.
 * @ingroup cpp_raii_api
 * @ingroup cpp_lowlevel_api
 *
 * Call after all plans are destroyed, before MPI_Finalize() for FFTW backend.
 * Safe to call multiple times.
 *
 * @return 0 on success, non-zero on error.
 */
[[nodiscard]] int finalize() noexcept;

// ---- Config wrapper classes ------------------------------------------------

/**
 * @brief RAII wrapper for N-D configuration objects.
 * @ingroup cpp_raii_api
 *
 * Manages the lifecycle of an shafft_nd_config_t C struct with RAII semantics.
 * Constructor performs init+resolve in one call. Provides operator->() for
 * direct arrow-syntax access to the underlying C struct fields.
 */
class ConfigND {
public:
  /**
   * @brief Construct, initialize, and resolve an N-D config.
   *
   * @param globalShape  Global extents per axis; ndim inferred from size().
   * @param precision    FFT type (C2C or Z2Z).
   * @param commDims     Process grid hint (empty = auto).
   * @param hintNda      Distributed axes hint (0 = auto).
   * @param strategy     Decomposition strategy.
   * @param outputPolicy Forward output-layout policy.
   * @param memLimit     Per-rank memory limit (0 = no limit).
   * @param comm         MPI communicator.
   * @throws None. Check status() after construction.
   */
  ConfigND(const std::vector<size_t>& globalShape,
           FFTType precision,
           const std::vector<int>& commDims = {},
           int hintNda = 0,
           DecompositionStrategy strategy = DecompositionStrategy::MAXIMIZE_NDA,
           TransformLayout outputPolicy = TransformLayout::REDISTRIBUTED,
           size_t memLimit = 0,
           MPI_Comm comm = MPI_COMM_WORLD) noexcept;

  /// @brief Non-copyable.
  ConfigND(const ConfigND&) = delete;
  ConfigND& operator=(const ConfigND&) = delete;

  /// @brief Move constructor.
  ConfigND(ConfigND&& other) noexcept;

  /// @brief Move assignment.
  ConfigND& operator=(ConfigND&& other) noexcept;

  /// @brief Destructor. Releases internal resources.
  ~ConfigND() noexcept;

  /// @brief Check if init succeeded.
  [[nodiscard]] int status() const noexcept { return status_; }

  /// @brief Read-only access to the underlying C struct.
  [[nodiscard]] const shafft_nd_config_t& cStruct() const noexcept { return cfg_; }

  /// @brief Mutable access (for advanced usage or direct field writes).
  [[nodiscard]] shafft_nd_config_t& cStruct() noexcept { return cfg_; }

  /// @brief Arrow-syntax access to underlying C struct (const).
  [[nodiscard]] const shafft_nd_config_t* operator->() const noexcept { return &cfg_; }

  /// @brief Arrow-syntax access to underlying C struct (mutable).
  [[nodiscard]] shafft_nd_config_t* operator->() noexcept { return &cfg_; }

  /// @brief Re-resolve configuration using stored worldComm.
  [[nodiscard]] int resolve() noexcept;

  /// @brief Check if the config has been successfully resolved.
  [[nodiscard]] bool isResolved() const noexcept {
    return (cfg_.flags & SHAFFT_CONFIG_RESOLVED) != 0;
  }

private:
  shafft_nd_config_t cfg_ = {};
  int status_ = -1;
};

/**
 * @brief RAII wrapper for 1-D configuration objects.
 * @ingroup cpp_raii_api
 *
 * Manages the lifecycle of an shafft_1d_config_t C struct with RAII semantics.
 * Constructor performs init+resolve in one call. Provides operator->() for
 * direct arrow-syntax access to the underlying C struct fields.
 */
class Config1D {
public:
  /**
   * @brief Construct, initialize, and resolve a 1-D config.
   * @param globalSize Global FFT length (> 0).
   * @param precision  FFT type (C2C or Z2Z).
   * @param comm       MPI communicator.
   * @throws None. Check status() after construction.
   */
  Config1D(size_t globalSize, FFTType precision, MPI_Comm comm) noexcept;

  /// @brief Non-copyable.
  Config1D(const Config1D&) = delete;
  Config1D& operator=(const Config1D&) = delete;

  /// @brief Move constructor.
  Config1D(Config1D&& other) noexcept;

  /// @brief Move assignment.
  Config1D& operator=(Config1D&& other) noexcept;

  /// @brief Destructor.
  ~Config1D() noexcept;

  /// @brief Check if init succeeded.
  [[nodiscard]] int status() const noexcept { return status_; }

  /// @brief Read-only access to the underlying C struct.
  [[nodiscard]] const shafft_1d_config_t& cStruct() const noexcept { return cfg_; }

  /// @brief Mutable access to the underlying C struct.
  [[nodiscard]] shafft_1d_config_t& cStruct() noexcept { return cfg_; }

  /// @brief Arrow-syntax access to underlying C struct (const).
  [[nodiscard]] const shafft_1d_config_t* operator->() const noexcept { return &cfg_; }

  /// @brief Arrow-syntax access to underlying C struct (mutable).
  [[nodiscard]] shafft_1d_config_t* operator->() noexcept { return &cfg_; }

  /// @brief Re-resolve configuration using stored worldComm.
  [[nodiscard]] int resolve() noexcept;

  /// @brief Check if resolved.
  [[nodiscard]] bool isResolved() const noexcept {
    return (cfg_.flags & SHAFFT_CONFIG_RESOLVED) != 0;
  }

private:
  shafft_1d_config_t cfg_ = {};
  int status_ = -1;
};

} // namespace shafft

#endif // SHAFFT_CPP_H
