/** @file shafft.h
 *  @brief C interface for SHAFFT.
 *  @ingroup c_api
 *
 *  Conventions:
 *  - Axis indices are 0-based.
 *  - Arrays noted as "length = ndim" must have at least @p ndim elements.
 *  - "ca" = indices of locally contiguous (non-distributed) axes,
 *    ordered innermost to outermost stride for the reported stage/layout.
 *  - Buffers may be swapped internally during execution; call
 *    ::shafftGetBuffers() after ::shafftExecute() to obtain the buffer
 *    that currently holds the transformed data.
 */

#ifndef SHAFFT_C_H
#define SHAFFT_C_H

#include <shafft/shafft_config.h>

#include <mpi.h>
#include <stddef.h>

#if SHAFFT_BACKEND_HIPFFT
#include <hip/hip_runtime_api.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if SHAFFT_BACKEND_HIPFFT
/**
 * @brief Set the HIP stream to use for all operations in the plan.
 * @ingroup c_api
 *
 * The plan must have been created and must not have in-flight work.
 * @param plan_ptr Plan pointer.
 * @param stream HIP stream to use.
 * @return 0 on success; non-zero on error.
 */
int shafftSetStream(void* plan_ptr, hipStream_t stream);
#endif

/**
 * @brief FFT type.
 * @ingroup c_api
 * - SHAFFT_C2C : single-precision complex-to-complex.
 * - SHAFFT_Z2Z : double-precision complex-to-complex.
 */
typedef enum { SHAFFT_C2C, SHAFFT_Z2Z } shafft_t;

/**
 * @brief Transform direction.
 * @ingroup c_api
 * - SHAFFT_FORWARD  : forward transform.
 * - SHAFFT_BACKWARD : backward transform.
 */
typedef enum {
  SHAFFT_FORWARD, /**< Forward transform. */
  SHAFFT_BACKWARD /**< Backward transform. */
} shafft_direction_t;

/**
 * @brief Tensor layout for querying local decomposition.
 * @ingroup c_api
 * - SHAFFT_TENSOR_LAYOUT_CURRENT     : layout associated with the last executed transform
 * direction.
 * - SHAFFT_TENSOR_LAYOUT_INITIAL     : layout before any transforms have been executed.
 * - SHAFFT_TENSOR_LAYOUT_TRANSFORMED : layout after a full forward or backward transform has been
 * executed.
 */
typedef enum {
  SHAFFT_TENSOR_LAYOUT_CURRENT,    /**< Current layout (forward or backward, depending on last
                                      execution). */
  SHAFFT_TENSOR_LAYOUT_INITIAL,    /**< Initial layout (before any transforms). */
  SHAFFT_TENSOR_LAYOUT_TRANSFORMED /**< Transformed layout (after execution). */
} shafft_tensor_layout_t;

/**
 * @brief Status/error codes returned by SHAFFT functions (aligned with shafft::Status).
 * @ingroup c_api
 * 0 indicates success; non-zero indicates an error condition.
 */
typedef enum {
  SHAFFT_SUCCESS = 0,             /**< Operation completed successfully. */
  SHAFFT_ERR_NULLPTR = 1,         /**< Null pointer argument. */
  SHAFFT_ERR_INVALID_COMM = 2,    /**< Invalid MPI communicator. */
  SHAFFT_ERR_NO_BUFFER = 3,       /**< Required data/work buffer was not set. */
  SHAFFT_ERR_PLAN_NOT_INIT = 4,   /**< Plan or subplan not initialized. */
  SHAFFT_ERR_INVALID_DIM = 5,     /**< Invalid dimension parameter. */
  SHAFFT_ERR_DIM_MISMATCH = 6,    /**< Dimension mismatch between arrays. */
  SHAFFT_ERR_INVALID_DECOMP = 7,  /**< Invalid or unsupported slab decomposition. */
  SHAFFT_ERR_INVALID_FFTTYPE = 8, /**< Invalid FFT type. */
  SHAFFT_ERR_ALLOC = 9,           /**< Memory allocation failed. */
  SHAFFT_ERR_BACKEND = 10,        /**< Backend (FFTW/hipFFT/HIP) failure. */
  SHAFFT_ERR_MPI = 11,            /**< MPI failure. */
  SHAFFT_ERR_INTERNAL = 12        /**< Uncategorized internal error. */
} shafft_status_t;

/**
 * @brief Error source domain for detailed diagnostics.
 * @ingroup c_api
 *
 * When a SHAFFT function returns an error, call shafft_last_error_source()
 * to determine which subsystem caused the failure.
 */
typedef enum {
  SHAFFT_ERRSRC_NONE = 0,   /**< No error or SHAFFT-internal error */
  SHAFFT_ERRSRC_MPI = 1,    /**< MPI library error (use MPI_Error_string) */
  SHAFFT_ERRSRC_HIP = 2,    /**< HIP runtime error (hipError_t) */
  SHAFFT_ERRSRC_HIPFFT = 3, /**< hipFFT library error (hipfftResult_t) */
  SHAFFT_ERRSRC_FFTW = 4,   /**< FFTW library error */
  SHAFFT_ERRSRC_SYSTEM = 5  /**< OS / allocation / errno-like errors */
} shafft_errsrc_t;

/**
 * @brief Get the SHAFFT status code from the last error on this thread.
 * @ingroup c_api
 * @return Status code (one of shafft_status_t values).
 */
int shafft_last_error_status(void);

/**
 * @brief Get the error source domain from the last error on this thread.
 * @ingroup c_api
 * @return Error source (one of shafft_errsrc_t values).
 */
int shafft_last_error_source(void);

/**
 * @brief Get the raw domain-specific error code from the last error.
 * @ingroup c_api
 *
 * The meaning of this code depends on shafft_last_error_source():
 * - SHAFFT_ERRSRC_MPI: MPI error code (use MPI_Error_string)
 * - SHAFFT_ERRSRC_HIP: hipError_t value
 * - SHAFFT_ERRSRC_HIPFFT: hipfftResult_t value
 * - SHAFFT_ERRSRC_FFTW: (not used, FFTW has no error codes)
 * - SHAFFT_ERRSRC_SYSTEM: errno or similar
 *
 * @return Raw error code from the underlying subsystem.
 */
int shafft_last_error_domain_code(void);

/**
 * @brief Get a human-readable message for the last error.
 * @ingroup c_api
 * @param buf    Buffer to receive the null-terminated message.
 * @param buflen Size of the buffer in bytes.
 * @return Number of characters written (excluding null terminator).
 */
int shafft_last_error_message(char* buf, int buflen);

/**
 * @brief Clear the last error state on this thread.
 * @ingroup c_api
 */
void shafft_clear_last_error(void);

/**
 * @brief Get the name of an error source as a string.
 * @ingroup c_api
 * @param source Error source value from shafft_last_error_source().
 * @return String name (e.g., "MPI", "HIP", "hipFFT", "FFTW").
 */
const char* shafft_error_source_name(int source);

/**
 * @brief Compute an N-dimensional slab decomposition with a specified number of distributed axes.
 * @ingroup c_api
 *
 * Determines the optimal Cartesian process grid and local tensor block for each
 * MPI rank based on the global tensor size, desired decomposition, and optional
 * memory constraints. Use the outputs to call ::shafftPlanNDA.
 *
 * @par Auto-selection mode (nda == 0 on input):
 * When @p nda is 0, the planner automatically selects the number of distributed
 * axes. The @p mem_limit parameter controls the selection strategy:
 * - `mem_limit > 0`: Maximize nda subject to per-rank memory staying under limit
 * - `mem_limit == 0`: Maximize nda (no memory constraint)
 * - `mem_limit < 0` (signed interpretation): Minimize nda (fewest distributed axes)
 *
 * @par Manual mode (nda > 0 on input):
 * When @p nda is positive, that exact value is used. The function fails if the
 * requested decomposition cannot be satisfied.
 *
 * @par Process grid computation:
 * The Cartesian process grid @p COMM_DIMS is computed automatically based on the
 * number of MPI ranks and tensor dimensions. The grid follows a "slab prefix"
 * structure: the first nda entries may be > 1, and all trailing entries are 1.
 * For example, with 8 ranks on a 64x64x32 tensor, COMM_DIMS might be [2,4,1].
 *
 * @par Per-axis caps:
 * Each COMM_DIMS[i] is capped by min(size[i], size[ndim-i-1]) to ensure valid
 * redistribution during the FFT computation.
 *
 * @par Inactive ranks:
 * If the tensor cannot be evenly distributed across all ranks, some ranks may
 * become inactive (receiving zero-sized local blocks). Inactive ranks are
 * handled gracefully: plan creation succeeds, execute() and normalize() become
 * no-ops, and a warning is printed to stderr.
 *
 * @param ndim        Number of tensor dimensions.
 * @param size        Global extents per axis (length = @p ndim).
 * @param nda         [in,out] Desired distributed axes on input (0 for auto);
 *                    actual value chosen by the planner on output.
 * @param subsize     [out] Local extents per axis for this rank (length = @p ndim).
 * @param offset      [out] Global starting indices per axis for this rank (length = @p ndim).
 * @param COMM_DIMS   [out] Cartesian process-grid dimensions (length = @p ndim).
 *                    Leading @p nda entries contain the grid; trailing entries are 1.
 * @param precision   FFT type: SHAFFT_C2C (single) or SHAFFT_Z2Z (double).
 * @param mem_limit   Per-rank memory limit in bytes (see auto-selection above).
 * @param c_comm      MPI communicator.
 * @return 0 on success; non-zero error code on failure.
 */
int shafftConfigurationNDA(int ndim, int* size, int* nda, int* subsize, int* offset, int* COMM_DIMS,
                           shafft_t precision, size_t mem_limit, MPI_Comm c_comm);

/**
 * @brief Compute a Cartesian decomposition and report communicator size.
 * @ingroup c_api
 *
 * Either validates a user-provided Cartesian process grid or auto-selects one,
 * then computes the local tensor block for each rank. Use the outputs to call
 * ::shafftPlanCart.
 *
 * @par Auto-selection mode (COMM_DIMS all zeros on input):
 * When all entries of @p COMM_DIMS are 0, the planner automatically selects
 * the optimal grid. The @p mem_limit parameter controls the strategy:
 * - `mem_limit >= 0`: Maximize number of distributed axes
 * - `mem_limit < 0`: Minimize number of distributed axes
 *
 * @par Manual mode (COMM_DIMS non-zero on input):
 * When @p COMM_DIMS contains non-zero values, the provided grid is validated
 * and used directly. The grid must follow the "slab prefix" structure:
 * - Leading entries (indices 0..d-1) must be > 1
 * - Trailing entries (indices d..ndim-1) must be 1 (or 0, normalized to 1)
 * - No gaps allowed (e.g., [2,1,4] is invalid)
 *
 * @par Grid constraints:
 * - Each COMM_DIMS[i] must not exceed min(size[i], size[ndim-i-1])
 * - The product of COMM_DIMS must not exceed the number of MPI ranks
 * - Single rank (world_size=1): COMM_DIMS must be all 1s
 *
 * @par COMM_SIZE output:
 * Set to the product of leading COMM_DIMS entries where COMM_DIMS[i] > 1.
 * This is the number of ranks that participate; remaining ranks are inactive.
 *
 * @par Inactive ranks:
 * Ranks with world_rank >= COMM_SIZE do not participate in the computation.
 * They are handled gracefully: plan creation succeeds, execute() and
 * normalize() become no-ops, and a warning is printed to stderr.
 *
 * @param ndim        Number of tensor dimensions.
 * @param size        Global extents per axis (length = @p ndim).
 * @param subsize     [out] Local extents per axis for this rank (length = @p ndim).
 * @param offset      [out] Global starting indices per axis for this rank (length = @p ndim).
 * @param COMM_DIMS   [in,out] Cartesian process-grid dimensions (length = @p ndim).
 *                    On input: zeros for auto-select, or explicit grid.
 *                    On output: the validated/chosen grid with trailing 1s.
 * @param COMM_SIZE   [out] Number of active ranks (product of leading grid dims).
 * @param precision   FFT type: SHAFFT_C2C (single) or SHAFFT_Z2Z (double).
 * @param mem_limit   Per-rank memory limit in bytes.
 * @param c_comm      MPI communicator.
 * @return 0 on success; non-zero error code on failure.
 */
int shafftConfigurationCart(int ndim, int* size, int* subsize, int* offset, int* COMM_DIMS,
                            int* COMM_SIZE, shafft_t precision, size_t mem_limit, MPI_Comm c_comm);

/**
 * @brief Allocate an uninitialized SHAFFT plan object.
 * @ingroup c_api
 *
 * On success, @p out_plan receives the plan pointer. Pass it to
 * ::shafftDestroy() to release resources.
 *
 * @param out_plan [out] Receives the newly allocated plan pointer (set to NULL on failure).
 * @return 0 on success;
 *         SHAFFT_ERR_NULLPTR if @p out_plan is NULL;
 *         SHAFFT_ERR_ALLOC if allocation fails.
 */
int shafftPlanCreate(void** out_plan);

/**
 * @brief Build a plan from an NDA decomposition.
 * @ingroup c_api
 *
 * See ::shafftConfigurationNDA.
 *
 * @param plan_ptr    Plan returned by ::shafftPlanCreate().
 * @param ndim        Global rank (number of axes).
 * @param nda         Number of distributed axes (NDA rank).
 * @param dimensions  Global extents per axis (length = @p ndim).
 * @param precision   FFT element/type (SHAFFT_C2C or SHAFFT_Z2Z).
 * @param c_comm      MPI communicator used by the plan.
 * @return 0 on success; non-zero on failure.
 */
int shafftPlanNDA(void* plan_ptr, int ndim, int nda, int dimensions[], shafft_t precision,
                  MPI_Comm c_comm);

/**
 * @brief Build a plan from an explicit Cartesian process grid.
 * @ingroup c_api
 *
 * See ::shafftConfigurationCart.
 *
 * @param plan_ptr    Plan returned by ::shafftPlanCreate().
 * @param ndim        Global rank (number of axes).
 * @param COMM_DIMS   Cartesian process-grid dimensions (length = number of distributed axes).
 * @param dimensions  Global extents per axis (length = @p ndim).
 * @param precision   FFT element/type (SHAFFT_C2C or SHAFFT_Z2Z).
 * @param c_comm      MPI communicator used by the plan.
 * @return 0 on success; non-zero on failure.
 */
int shafftPlanCart(void* plan_ptr, int ndim, int COMM_DIMS[], int dimensions[], shafft_t precision,
                   MPI_Comm c_comm);

/**
 * @brief Destroy a plan and release resources.
 * @ingroup c_api
 *
 * Does not free user-provided data/work buffers.
 *
 * @param plan_ptr plan pointer.
 * @return 0 on success; non-zero on failure.
 */
int shafftDestroy(void** plan_ptr);

/**
 * @brief Query the local layout for the current plan.
 * @ingroup c_api
 *
 * @param plan_ptr Plan pointer.
 * @param subsize  [out] Local extents per axis for this rank (length = plan rank).
 * @param offset   [out] Global starting indices per axis for this rank (length = plan rank).
 * @param layout   Layout to query (SHAFFT_TENSOR_LAYOUT_CURRENT, SHAFFT_TENSOR_LAYOUT_INITIAL, or
 * SHAFFT_TENSOR_LAYOUT_TRANSFORMED).
 * @return 0 on success; non-zero on failure.
 */
int shafftGetLayout(void* plan_ptr, int* subsize, int* offset, shafft_tensor_layout_t layout);

/**
 * @brief Query the locally contiguous and distributed axes for the current plan.
 * @ingroup c_api
 *
 * @param plan_ptr Plan pointer.
 * @param ca       [out] Indices of locally contiguous (non-distributed) axes,
 *                 ordered innermost to outermost stride for the reported layout
 *                 (length = number of contiguous axes for the reported layout).
 * @param da       [out] Indices of distributed axes, ordered innermost to
 *                 outermost stride for the reported layout
 *                 (length = number of distributed axes for the reported layout).
 * @param layout   Layout to query (SHAFFT_TENSOR_LAYOUT_CURRENT, SHAFFT_TENSOR_LAYOUT_INITIAL, or
 * SHAFFT_TENSOR_LAYOUT_TRANSFORMED).
 * @return 0 on success; non-zero on failure.
 */
int shafftGetAxes(void* plan_ptr, int* ca, int* da, shafft_tensor_layout_t layout);

/**
 * @brief Report the total buffer size required by the plan (in elements).
 * @ingroup c_api
 *
 * Returns the number of complex elements the plan expects to have available
 * in the attached data and work buffers.
 *
 * @param plan       Plan pointer.
 * @param alloc_size [out] Required element count; 0 on error.
 * @return 0 on success; non-zero on failure.
 */
int shafftGetAllocSize(void* plan, size_t* alloc_size);

/**
 * @brief Attach data and work buffers to a plan.
 * @ingroup c_api
 *
 * The plan will use these buffers for global redistributions and FFT kernels.
 * The caller allocates and owns the buffers. Buffers must remain valid for the
 * lifetime of the plan or until new buffers are set.
 *
 * Errors:
 * - SHAFFT_ERR_NULLPTR     : @p plan_ptr, @p data, or @p work is NULL.
 * - SHAFFT_ERR_PLAN_NOT_INIT : Plan has not been initialized.
 *
 * @param plan_ptr Plan pointer.
 * @param data     Pointer to the main data buffer.
 * @param work     Pointer to the work/scratch buffer.
 * @return 0 on success; non-zero status on failure (see Errors).
 */
int shafftSetBuffers(void* plan_ptr, void* data, void* work);

/**
 * @brief Retrieve the currently attached buffer pointers.
 * @ingroup c_api
 *
 * Note: During execution the library may swap the roles of @p data and @p work.
 * Call this after ::shafftExecute() to locate the buffer that holds the
 * transformed data.
 *
 * Errors:
 * - SHAFFT_ERR_NULLPTR       : @p plan_ptr, @p data, or @p work is NULL.
 * - SHAFFT_ERR_PLAN_NOT_INIT : Plan has not been initialized.
 * - SHAFFT_ERR_NO_BUFFER     : No buffers have been attached.
 *
 * @param plan_ptr Plan pointer.
 * @param data     [out] Receives current data buffer pointer.
 * @param work     [out] Receives current work buffer pointer.
 * @return 0 on success; non-zero status on failure (see Errors).
 */
int shafftGetBuffers(void* plan_ptr, void** data, void** work);

/**
 * @brief Execute the FFT associated with the plan.
 * @ingroup c_api
 *
 * Performs the forward or backward transform on the attached buffers according
 * to the plan configuration.
 *
 * @param plan_ptr  Plan pointer.
 * @param direction Transform direction (SHAFFT_FORWARD or SHAFFT_BACKWARD).
 * @return 0 on success; non-zero on failure.
 */
int shafftExecute(void* plan_ptr, shafft_direction_t direction);

/**
 * @brief Normalize the current data buffer.
 * @ingroup c_api
 *
 * @param plan_ptr Plan pointer.
 * @return 0 on success; non-zero on failure.
 */
int shafftNormalize(void* plan_ptr);

//------------------------------------------------------------------------------
// Portable memory allocation helpers
//------------------------------------------------------------------------------

/**
 * @brief Allocate a single-precision complex buffer suitable for the current backend.
 * @ingroup c_api
 *
 * Allocates device memory on GPU backends (hipMalloc), host memory on CPU backends.
 * Use shafftFreeBufferF() to release.
 *
 * @param count Number of complex elements to allocate.
 * @param buf   [out] Receives the allocated buffer pointer.
 * @return 0 on success; non-zero on failure.
 */
int shafftAllocBufferF(size_t count, void** buf);

/**
 * @brief Allocate a double-precision complex buffer suitable for the current backend.
 * @ingroup c_api
 *
 * Allocates device memory on GPU backends (hipMalloc), host memory on CPU backends.
 * Use shafftFreeBufferD() to release.
 *
 * @param count Number of complex elements to allocate.
 * @param buf   [out] Receives the allocated buffer pointer.
 * @return 0 on success; non-zero on failure.
 */
int shafftAllocBufferD(size_t count, void** buf);

/**
 * @brief Free a single-precision buffer allocated with shafftAllocBufferF().
 * @ingroup c_api
 *
 * @param buf Buffer to free (may be NULL).
 * @return 0 on success; non-zero on failure.
 */
int shafftFreeBufferF(void* buf);

/**
 * @brief Free a double-precision buffer allocated with shafftAllocBufferD().
 * @ingroup c_api
 *
 * @param buf Buffer to free (may be NULL).
 * @return 0 on success; non-zero on failure.
 */
int shafftFreeBufferD(void* buf);

//------------------------------------------------------------------------------
// Portable memory copy helpers
//------------------------------------------------------------------------------

/**
 * @brief Copy single-precision complex data from host memory to a SHAFFT buffer.
 * @ingroup c_api
 *
 * On GPU backends, performs hipMemcpy (host-to-device).
 * On CPU backends, performs memcpy.
 *
 * @param dst   Destination buffer (allocated via shafftAllocBufferF or user-managed).
 * @param src   Source host memory.
 * @param count Number of complex elements to copy.
 * @return 0 on success; non-zero on failure.
 */
int shafftCopyToBufferF(void* dst, const void* src, size_t count);

/**
 * @brief Copy double-precision complex data from host memory to a SHAFFT buffer.
 * @ingroup c_api
 *
 * On GPU backends, performs hipMemcpy (host-to-device).
 * On CPU backends, performs memcpy.
 *
 * @param dst   Destination buffer (allocated via shafftAllocBufferD or user-managed).
 * @param src   Source host memory.
 * @param count Number of complex elements to copy.
 * @return 0 on success; non-zero on failure.
 */
int shafftCopyToBufferD(void* dst, const void* src, size_t count);

/**
 * @brief Copy single-precision complex data from a SHAFFT buffer to host memory.
 * @ingroup c_api
 *
 * On GPU backends, performs hipMemcpy (device-to-host).
 * On CPU backends, performs memcpy.
 *
 * @param dst   Destination host memory.
 * @param src   Source buffer.
 * @param count Number of complex elements to copy.
 * @return 0 on success; non-zero on failure.
 */
int shafftCopyFromBufferF(void* dst, const void* src, size_t count);

/**
 * @brief Copy double-precision complex data from a SHAFFT buffer to host memory.
 * @ingroup c_api
 *
 * On GPU backends, performs hipMemcpy (device-to-host).
 * On CPU backends, performs memcpy.
 *
 * @param dst   Destination host memory.
 * @param src   Source buffer.
 * @param count Number of complex elements to copy.
 * @return 0 on success; non-zero on failure.
 */
int shafftCopyFromBufferD(void* dst, const void* src, size_t count);

//------------------------------------------------------------------------------
// Library information
//------------------------------------------------------------------------------

/**
 * @brief Get the name of the FFT backend used at compile time.
 * @ingroup c_api
 * @return "FFTW" or "hipFFT".
 */
const char* shafftGetBackendName(void);

/**
 * @brief Get the library version as major, minor, patch components.
 * @ingroup c_api
 *
 * @param major [out] Receives major version.
 * @param minor [out] Receives minor version.
 * @param patch [out] Receives patch version.
 */
void shafftGetVersion(int* major, int* minor, int* patch);

/**
 * @brief Get the library version as a string (e.g., "0.1.0-alpha").
 * @ingroup c_api
 * @return Version string.
 */
const char* shafftGetVersionString(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // SHAFFT_C_H
