/** @file shafft.hpp
 *  @brief C++ interface for SHAFFT.
 *  @ingroup cpp_api
 */

#ifndef SHAFFT_CPP_H
#define SHAFFT_CPP_H

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

namespace shafft {

//==============================================================================
// RAII Plan class (recommended for C++ users)
//==============================================================================

/**
 * @brief RAII wrapper for SHAFFT distributed FFT plans.
 *
 * The Plan class provides automatic resource management and a cleaner object-oriented
 * interface for distributed FFT operations. Resources are automatically released
 * when the Plan goes out of scope.
 *
 * @note This class does NOT manage user-provided data/work buffers - those must be
 *       freed separately using freeBuffer() or user's own memory management.
 *
 * Example usage:
 * @code
 *   shafft::Plan plan;
 *   if (plan.init(1, {64, 64, 32}, shafft::FFTType::C2C, MPI_COMM_WORLD) != 0) {
 *     // handle error
 *   }
 *
 *   shafft::complexf *data, *work;
 *   shafft::allocBuffer(plan.allocSize(), &data);
 *   shafft::allocBuffer(plan.allocSize(), &work);
 *
 *   plan.setBuffers(data, work);
 *   plan.execute(shafft::FFTDirection::FORWARD);
 *   plan.normalize();
 * @endcode
 */
class Plan {
 public:
  /// @brief Default constructor. Creates an uninitialized plan.
  Plan() noexcept = default;

  /// @brief Destructor. Releases all internal resources.
  ~Plan() noexcept;

  /**
   * @brief Explicitly release all internal resources.
   *
   * Call this before MPI_Finalize() if the Plan outlives MPI, otherwise
   * the destructor will attempt MPI operations after MPI is finalized.
   * After calling release(), the plan is in an uninitialized state.
   */
  void release() noexcept;

  /// @brief Move constructor.
  Plan(Plan&& other) noexcept;

  /// @brief Move assignment operator.
  Plan& operator=(Plan&& other) noexcept;

  // Non-copyable
  Plan(const Plan&) = delete;
  Plan& operator=(const Plan&) = delete;

  //----------------------------------------------------------------------------
  // Initialization
  //----------------------------------------------------------------------------

  /**
   * @brief Initialize plan with NDA (N Distributed Axes) decomposition.
   *
   * @param nda        Number of distributed axes (typically 1 or 2).
   * @param dimensions Global tensor dimensions.
   * @param type       FFT type (C2C for single precision, Z2Z for double).
   * @param comm       MPI communicator.
   * @return Status code (0 on success).
   */
  [[nodiscard]] int init(int nda, const std::vector<int>& dimensions, FFTType type,
                         MPI_Comm comm) noexcept;

  /**
   * @brief Initialize plan with explicit Cartesian process grid.
   *
   * @param commDims   Process grid dimensions.
   * @param dimensions Global tensor dimensions.
   * @param type       FFT type.
   * @param comm       MPI communicator.
   * @return Status code (0 on success).
   */
  [[nodiscard]] int initCart(const std::vector<int>& commDims, const std::vector<int>& dimensions,
                             FFTType type, MPI_Comm comm) noexcept;

  //----------------------------------------------------------------------------
  // Buffer management
  //----------------------------------------------------------------------------

  /**
   * @brief Attach data and work buffers to the plan.
   * @return Status code (0 on success).
   */
  [[nodiscard]] int setBuffers(complexf* data, complexf* work) noexcept;
  [[nodiscard]] int setBuffers(complexd* data, complexd* work) noexcept;

  /**
   * @brief Retrieve current data and work buffer pointers.
   *
   * After execute(), buffers may be swapped internally.
   * @return Status code (0 on success).
   */
  [[nodiscard]] int getBuffers(complexf** data, complexf** work) noexcept;
  [[nodiscard]] int getBuffers(complexd** data, complexd** work) noexcept;

#if SHAFFT_BACKEND_HIPFFT
  [[nodiscard]] int setBuffers(hipFloatComplex* data, hipFloatComplex* work) noexcept;
  [[nodiscard]] int setBuffers(hipDoubleComplex* data, hipDoubleComplex* work) noexcept;
  [[nodiscard]] int getBuffers(hipFloatComplex** data, hipFloatComplex** work) noexcept;
  [[nodiscard]] int getBuffers(hipDoubleComplex** data, hipDoubleComplex** work) noexcept;
  [[nodiscard]] int setStream(hipStream_t stream) noexcept;
#endif
#if SHAFFT_BACKEND_FFTW
  [[nodiscard]] int setBuffers(fftw_complex* data, fftw_complex* work) noexcept;
  [[nodiscard]] int setBuffers(fftwf_complex* data, fftwf_complex* work) noexcept;
  [[nodiscard]] int getBuffers(fftw_complex** data, fftw_complex** work) noexcept;
  [[nodiscard]] int getBuffers(fftwf_complex** data, fftwf_complex** work) noexcept;
#endif

  //----------------------------------------------------------------------------
  // Execution
  //----------------------------------------------------------------------------

  /**
   * @brief Execute the FFT.
   * @param direction FORWARD or BACKWARD.
   * @return Status code (0 on success).
   */
  [[nodiscard]] int execute(FFTDirection direction) noexcept;

  /**
   * @brief Apply normalization to the transformed data.
   *
   * Normalizes by `1/sqrt(N)` for each FFT that was executed, where N is the
   * total tensor size. After a forward-backward pair, calling `normalize()`
   * returns data to its original scale.
   *
   * @return Status code (0 on success).
   */
  [[nodiscard]] int normalize() noexcept;

  //----------------------------------------------------------------------------
  // Queries
  //----------------------------------------------------------------------------

  /**
   * @brief Get the required buffer allocation size (in elements).
   * @return Number of elements needed for data/work buffers.
   */
  [[nodiscard]] size_t allocSize() const noexcept;

  /**
   * @brief Query the local tensor layout.
   * @param[out] subsize Local dimensions per axis.
   * @param[out] offset  Global offset per axis.
   * @param layout Which layout to query (CURRENT, INITIAL, or TRANSFORMED).
   * @return Status code (0 on success).
   */
  [[nodiscard]] int getLayout(std::vector<int>& subsize, std::vector<int>& offset,
                              TensorLayout layout = TensorLayout::CURRENT) const noexcept;

  /**
   * @brief Query the local tensor axes distribution.
   * @param[out] ca Contiguous (non-distributed) axes.
   * @param[out] da Distributed axes.
   * @param layout Which layout to query.
   * @return Status code (0 on success).
   */
  [[nodiscard]] int getAxes(std::vector<int>& ca, std::vector<int>& da,
                            TensorLayout layout = TensorLayout::CURRENT) const noexcept;

  /// @brief Check if the plan has been initialized.
  [[nodiscard]] bool isInitialized() const noexcept { return data_ != nullptr; }

  /// @brief Check if the plan is valid (alias for isInitialized).
  explicit operator bool() const noexcept { return isInitialized(); }

  /**
   * @brief Check if this rank is active (has work to do).
   *
   * Inactive ranks do not participate in the FFT computation. This can occur
   * when the tensor size or decomposition does not evenly divide across all
   * MPI ranks. Inactive ranks can still call execute() and normalize() safely;
   * these become no-ops.
   *
   * @return true if this rank participates in the FFT, false if excluded.
   */
  [[nodiscard]] bool isActive() const noexcept;

  /// @brief Get the underlying PlanData pointer (for advanced use/interop).
  [[nodiscard]] PlanData* data() noexcept { return data_; }
  [[nodiscard]] const PlanData* data() const noexcept { return data_; }

 private:
  PlanData* data_ = nullptr;
};

//==============================================================================
// Free functions (C-style API for backward compatibility and C interop)
//==============================================================================

#if SHAFFT_BACKEND_HIPFFT
/**
 * @brief Set the HIP stream for a SHAFFT plan.
 * @ingroup cpp_api
 *
 * @param plan   Plan handle (initialized).
 * @param stream HIP stream to set.
 * @return Status code (0 on success, non-zero on failure).
 */
int setStream(PlanData* plan, hipStream_t stream);

/// @brief Retrieve buffers (hipFloatComplex overload).
/// @ingroup cpp_api
int getBuffers(PlanData* plan, hipFloatComplex** data, hipFloatComplex** work) noexcept;
/// @brief Retrieve buffers (hipDoubleComplex overload).
/// @ingroup cpp_api
int getBuffers(PlanData* plan, hipDoubleComplex** data, hipDoubleComplex** work) noexcept;
/// @brief Set buffers (hipFloatComplex overload).
/// @ingroup cpp_api
int setBuffers(PlanData* plan, hipFloatComplex* data, hipFloatComplex* work) noexcept;
/// @brief Set buffers (hipDoubleComplex overload).
/// @ingroup cpp_api
int setBuffers(PlanData* plan, hipDoubleComplex* data, hipDoubleComplex* work) noexcept;
#endif
#if SHAFFT_BACKEND_FFTW
/// @brief Retrieve buffers (fftw_complex overload).
/// @ingroup cpp_api
int getBuffers(PlanData* plan, fftw_complex** data, fftw_complex** work) noexcept;
/// @brief Retrieve buffers (fftwf_complex overload).
/// @ingroup cpp_api
int getBuffers(PlanData* plan, fftwf_complex** data, fftwf_complex** work) noexcept;
/// @brief Set buffers (fftw_complex overload).
/// @ingroup cpp_api
int setBuffers(PlanData* plan, fftw_complex* data, fftw_complex* work) noexcept;
/// @brief Set buffers (fftwf_complex overload).
/// @ingroup cpp_api
int setBuffers(PlanData* plan, fftwf_complex* data, fftwf_complex* work) noexcept;
#endif

//------------------------------------------------------------------------------
// Backend-agnostic buffer functions (portable across CPU and GPU)
//------------------------------------------------------------------------------

/**
 * @brief Attach data and work buffers using portable complex types.
 * @ingroup cpp_api
 *
 * Works identically on CPU (FFTW) and GPU (HIPFFT) backends.
 * On GPU backends, buffers must reside in device memory.
 * On CPU backends, buffers must reside in host memory.
 *
 * @param plan Plan handle (initialized).
 * @param data Data buffer pointer.
 * @param work Work/scratch buffer pointer.
 * @return Status code (0 on success, non-zero on failure).
 */
int setBuffers(PlanData* plan, complexf* data, complexf* work) noexcept;
int setBuffers(PlanData* plan, complexd* data, complexd* work) noexcept;

/**
 * @brief Retrieve the current data and work buffer pointers.
 * @ingroup cpp_api
 *
 * After execute(), buffers may be swapped; use this to obtain
 * the pointer that currently holds the transformed data.
 *
 * @param plan Plan handle (initialized).
 * @param data [out] Receives current data buffer pointer.
 * @param work [out] Receives current work buffer pointer.
 * @return Status code (0 on success, non-zero on failure).
 */
int getBuffers(PlanData* plan, complexf** data, complexf** work) noexcept;
int getBuffers(PlanData* plan, complexd** data, complexd** work) noexcept;

//------------------------------------------------------------------------------
// Portable memory allocation helpers
//------------------------------------------------------------------------------

/**
 * @brief Allocate a buffer suitable for the current backend.
 * @ingroup cpp_api
 *
 * Allocates device memory on GPU backends (hipMalloc), host memory on CPU backends.
 * Use freeBuffer() to release.
 *
 * @param count Number of elements to allocate.
 * @param buf   [out] Receives the allocated buffer pointer.
 * @return Status code (0 on success, non-zero on failure).
 */
int allocBuffer(size_t count, complexf** buf) noexcept;
int allocBuffer(size_t count, complexd** buf) noexcept;

/**
 * @brief Free a buffer allocated with allocBuffer().
 * @ingroup cpp_api
 *
 * @param buf Buffer to free (may be nullptr).
 * @return Status code (0 on success, non-zero on failure).
 */
int freeBuffer(complexf* buf) noexcept;
int freeBuffer(complexd* buf) noexcept;

//------------------------------------------------------------------------------
// Portable memory copy helpers
//------------------------------------------------------------------------------

/**
 * @brief Copy data from host memory to a SHAFFT buffer.
 * @ingroup cpp_api
 *
 * On GPU backends, performs hipMemcpy (host-to-device).
 * On CPU backends, performs std::memcpy.
 *
 * @param dst   Destination buffer (allocated via allocBuffer or user-managed).
 * @param src   Source host memory.
 * @param count Number of elements to copy.
 * @return Status code (0 on success, non-zero on failure).
 */
int copyToBuffer(complexf* dst, const complexf* src, size_t count) noexcept;
int copyToBuffer(complexd* dst, const complexd* src, size_t count) noexcept;

/**
 * @brief Copy data from a SHAFFT buffer to host memory.
 * @ingroup cpp_api
 *
 * On GPU backends, performs hipMemcpy (device-to-host).
 * On CPU backends, performs std::memcpy.
 *
 * @param dst   Destination host memory.
 * @param src   Source buffer.
 * @param count Number of elements to copy.
 * @return Status code (0 on success, non-zero on failure).
 */
int copyFromBuffer(complexf* dst, const complexf* src, size_t count) noexcept;
int copyFromBuffer(complexd* dst, const complexd* src, size_t count) noexcept;

/**
 * @brief Create a new SHAFFT plan (C-style API).
 * @ingroup cpp_api
 *
 * Allocates (with nothrow) a plan object. Destroy with destroy().
 * @note Prefer using the RAII shafft::Plan class for automatic resource management.
 *
 * @param out  [out] Receives the newly created plan pointer.
 * @return Status code (0 on success, non-zero on failure).
 *         On allocation failure returns SHAFFT_ERR_ALLOC and sets out to nullptr.
 */
int planCreate(PlanData** out);

/**
 * @brief Compute an NDA (prefix) slab decomposition with a desired number of distributed axes.
 * @ingroup cpp_api
 *
 * This function determines the optimal Cartesian process grid and local tensor
 * block for each MPI rank based on the global tensor size, desired decomposition,
 * and optional memory constraints.
 *
 * @par Auto-selection mode (nda == 0 on input):
 * When @p nda is 0, the planner automatically selects the number of distributed
 * axes. The `mem_limit` parameter controls the selection strategy:
 * - `mem_limit > 0`: Maximize nda subject to per-rank memory staying under limit
 * - `mem_limit == 0`: Maximize nda (no memory constraint)
 * - `mem_limit < 0` (signed interpretation): Minimize nda (fewest distributed axes)
 *
 * @par Manual mode (nda > 0 on input):
 * When @p nda is positive, that exact value is used. The function fails if the
 * requested decomposition cannot be satisfied.
 *
 * @par Process grid computation:
 * The Cartesian process grid `COMM_DIMS` is computed automatically based on the
 * number of MPI ranks and tensor dimensions. The grid follows a "slab prefix"
 * structure: the first `nda` entries may be > 1, and all trailing entries are 1.
 * For example, with 8 ranks on a 64×64×32 tensor, COMM_DIMS might be [2,4,1].
 *
 * @par Per-axis caps:
 * Each COMM_DIMS[i] is capped by min(size[i], size[ndim-i-1]) to ensure valid
 * redistribution during the FFT computation.
 *
 * @par Inactive ranks:
 * If the tensor cannot be evenly distributed across all ranks, some ranks may
 * become inactive (receiving zero-sized local blocks). Inactive ranks are
 * handled gracefully: plan creation succeeds, execute() and normalize() become
 * no-ops, and isActive() returns false. A warning is printed to stderr when
 * a rank becomes inactive.
 *
 * @param size       Global tensor extents per axis.
 * @param nda        [in,out] Desired distributed axes on input; actual value on output.
 * @param subsize    [out] Local extents per axis for this rank.
 * @param offset     [out] Global starting indices per axis for this rank.
 * @param COMM_DIMS  [out] Cartesian process-grid dimensions (length = ndim).
 *                   Leading `nda` entries contain the grid; trailing entries are 1.
 * @param precision  FFT type (C2C for single, Z2Z for double precision).
 * @param mem_limit  Per-rank memory limit in bytes (see auto-selection above).
 * @param COMM       MPI communicator.
 * @return Status code (SHAFFT_SUCCESS on success, error code on failure).
 */
int configurationNDA(const std::vector<int>& size, int& nda, std::vector<int>& subsize,
                     std::vector<int>& offset, std::vector<int>& COMM_DIMS, FFTType precision,
                     size_t mem_limit, MPI_Comm COMM);

/**
 * @brief Compute/validate a Cartesian decomposition and report communicator size.
 * @ingroup cpp_api
 *
 * This function either validates a user-provided Cartesian process grid or
 * auto-selects one, then computes the local tensor block for each rank.
 *
 * @par Auto-selection mode (COMM_DIMS all zeros on input):
 * When all entries of @p COMM_DIMS are 0, the planner automatically selects
 * the optimal grid. The `mem_limit` parameter controls the strategy:
 * - `mem_limit >= 0`: Maximize number of distributed axes
 * - `mem_limit < 0`: Minimize number of distributed axes
 *
 * @par Manual mode (COMM_DIMS non-zero on input):
 * When @p COMM_DIMS contains non-zero values, the provided grid is validated
 * and used directly. The grid must follow the "slab prefix" structure:
 * - Leading entries (indices 0..d-1) must be > 1
 * - Trailing entries (indices d..ndim-1) must be 1 (or 0, which is normalized to 1)
 * - No gaps are allowed (e.g., [2,1,4] is invalid)
 *
 * @par Grid constraints:
 * - Each COMM_DIMS[i] must not exceed min(size[i], size[ndim-i-1])
 * - The product of COMM_DIMS must not exceed the number of MPI ranks
 * - Single rank (world_size=1): COMM_DIMS must be all 1s
 *
 * @par COMM_SIZE output:
 * The @p COMM_SIZE output is set to the product of the leading COMM_DIMS entries
 * where COMM_DIMS[i] > 1. This is the number of ranks that will participate in
 * the computation; remaining ranks become inactive.
 *
 * @par Inactive ranks:
 * Ranks with world_rank >= COMM_SIZE do not participate in the computation.
 * They are handled gracefully: plan creation succeeds, execute() and
 * normalize() become no-ops, isActive() returns false, and allocSize()
 * returns 0. A warning is printed to stderr when a rank becomes inactive.
 *
 * @param size       Global tensor extents per axis.
 * @param subsize    [out] Local extents per axis for this rank.
 * @param offset     [out] Global starting indices per axis for this rank.
 * @param COMM_DIMS  [in,out] Cartesian process-grid dimensions.
 *                   On input: zeros for auto-select, or explicit grid.
 *                   On output: the validated/chosen grid with trailing 1s.
 * @param COMM_SIZE  [out] Number of active ranks (product of leading grid dims).
 * @param precision  FFT type (C2C for single, Z2Z for double precision).
 * @param mem_limit  Per-rank memory limit in bytes.
 * @param COMM       MPI communicator.
 * @return Status code (SHAFFT_SUCCESS on success, error code on failure).
 */
int configurationCart(const std::vector<int>& size, std::vector<int>& subsize,
                      std::vector<int>& offset, std::vector<int>& COMM_DIMS, int& COMM_SIZE,
                      FFTType precision, size_t mem_limit, MPI_Comm COMM);

/**
 * @brief Build a plan from an NDA decomposition (C-style API).
 * @ingroup cpp_api
 * @note Prefer using Plan::init() for the RAII interface.
 */
int planNDA(PlanData* plan, int nda, const std::vector<int>& dimensions, FFTType precision,
            MPI_Comm COMM);

/**
 * @brief Build a plan from an explicit Cartesian process grid (C-style API).
 * @ingroup cpp_api
 * @note Prefer using Plan::initCart() for the RAII interface.
 */
int planCart(PlanData* plan, const std::vector<int>& COMM_DIMS, const std::vector<int>& dimensions,
             FFTType precision, MPI_Comm COMM);

/**
 * @brief Release resources held by the plan and null out the pointer.
 * @ingroup cpp_api
 *
 * Does not free user-provided data/work buffers.
 * @note The RAII Plan class handles this automatically in its destructor.
 */
int destroy(PlanData** plan);

/**
 * @brief Query the current/initial/transformed tensor layout.
 * @ingroup cpp_api
 *
 * @return Status::SHAFFT_SUCCESS on success, error otherwise.
 */
int getLayout(const PlanData* plan, std::vector<int>& subsize, std::vector<int>& offset,
              TensorLayout layout);

/**
 * @brief Query the current/initial/transformed tensor axes (contiguous/distributed).
 * @ingroup cpp_api
 *
 * @return Status::SHAFFT_SUCCESS on success, error otherwise.
 */
int getAxes(const PlanData* plan, std::vector<int>& ca, std::vector<int>& da, TensorLayout layout);

/**
 * @brief Report the total buffer size required by the plan (in elements).
 * @ingroup cpp_api
 *
 * Fills @p alloc_size with the number of elements (of the plan's FFT type)
 * required across data + work buffers.
 *
 * @param plan        Plan handle (initialized).
 * @param alloc_size  [out] Required element count.
 * @return Status code (0 on success, non-zero on failure).
 */
int getAllocSize(const PlanData* plan, size_t& alloc_size);

/**
 * @brief Execute the FFT associated with the plan.
 * @ingroup cpp_api
 */
int execute(PlanData* plan, FFTDirection direction);

/**
 * @brief Apply normalization to the current data buffer.
 * @ingroup cpp_api
 */
int normalize(PlanData* plan);

//==============================================================================
// Library information
//==============================================================================

/**
 * @brief Get the name of the FFT backend used at compile time.
 * @ingroup cpp_api
 * @return "FFTW" or "hipFFT".
 */
inline const char* getBackendName() noexcept {
  return SHAFFT_BACKEND_NAME;
}

/**
 * @brief Library version information.
 * @ingroup cpp_api
 */
struct Version {
  int major;  ///< Major version number.
  int minor;  ///< Minor version number.
  int patch;  ///< Patch version number.
};

/**
 * @brief Get the library version as a struct.
 * @ingroup cpp_api
 */
inline Version getVersion() noexcept {
  return {SHAFFT_VERSION_MAJOR, SHAFFT_VERSION_MINOR, SHAFFT_VERSION_PATCH};
}

/**
 * @brief Get the library version as a string (e.g., "0.1.0-alpha").
 * @ingroup cpp_api
 */
inline const char* getVersionString() noexcept {
  // Computed at compile time
  static const char version[] = SHAFFT_STRINGIFY(SHAFFT_VERSION_MAJOR) "." SHAFFT_STRINGIFY(
      SHAFFT_VERSION_MINOR) "." SHAFFT_STRINGIFY(SHAFFT_VERSION_PATCH) SHAFFT_VERSION_SUFFIX;
  return version;
}

}  // namespace shafft

#endif  // SHAFFT_CPP_H
