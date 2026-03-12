/** @brief C interface for SHAFFT.
 *  @ingroup c_api
 *
 *  Conventions:
 *  - Axis indices are 0-based.
 *  - Arrays noted as "length = ndim" must have at least @p ndim elements.
 *  - "ca" = indices of locally contiguous (non-distributed) axes,
 *    ordered innermost to outermost stride for the reported stage/layout.
 *  - Buffers may be swapped internally during execution; call
 *    ::shafftGetBuffers() after ::shafftExecute() to obtain the buffer
 *    that currently holds the most recent execution result.
 */

#ifndef SHAFFT_C_H
#define SHAFFT_C_H

#include <shafft/shafft_config.h>

#include <mpi.h>
#include <stddef.h> // NOLINT(modernize-deprecated-headers)

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
 * Works on both N-D and 1D plans. The plan must have been created
 * and must not have in-flight work.
 *
 * @param planPtr Plan pointer (N-D or 1D).
 * @param stream   HIP stream to use.
 * @return 0 on success; non-zero on error.
 */
int shafftSetStream(void* planPtr, hipStream_t stream);
#endif

/**
 * @brief FFT type.
 * @ingroup c_api
 * - SHAFFT_C2C : single-precision complex-to-complex.
 * - SHAFFT_Z2Z : double-precision complex-to-complex.
 */
// NOLINTNEXTLINE(modernize-use-using)
typedef enum { SHAFFT_C2C, SHAFFT_Z2Z } shafft_t;

/**
 * @brief Transform direction.
 * @ingroup c_api
 * - SHAFFT_FORWARD  : forward transform.
 * - SHAFFT_BACKWARD : backward transform.
 */
// NOLINTNEXTLINE(modernize-use-using)
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
 * - SHAFFT_TENSOR_LAYOUT_REDISTRIBUTED : user-visible post-forward redistributed layout
 * (policy-dependent).
 */
// NOLINTNEXTLINE(modernize-use-using)
typedef enum {
  SHAFFT_TENSOR_LAYOUT_CURRENT,      /**< Current layout (forward or backward, depending on last
                                        execution). */
  SHAFFT_TENSOR_LAYOUT_INITIAL,      /**< Initial layout (before any transforms). */
  SHAFFT_TENSOR_LAYOUT_REDISTRIBUTED /**< User-visible redistributed layout state. */
} shafft_tensor_layout_t;

/**
 * @brief Forward transform output policy for planning.
 * @ingroup c_api
 *
 * Controls the data layout after a forward transform for N-D plans.
 * Used by shafftNDInit(); not applicable to shafft1DInit().
 */
// NOLINTNEXTLINE(modernize-use-using)
typedef enum {
  SHAFFT_LAYOUT_REDISTRIBUTED = 0, /**< Keep post-forward redistributed layout. */
  SHAFFT_LAYOUT_INITIAL = 1        /**< Restore initial layout after forward transform. */
} shafft_transform_layout_t;

/**
 * @brief Status/error codes returned by SHAFFT functions (aligned with shafft::Status).
 * @ingroup c_api
 * 0 indicates success; non-zero indicates an error condition.
 */
// NOLINTNEXTLINE(modernize-use-using)
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
  SHAFFT_ERR_INVALID_LAYOUT = 12, /**< Layout parameters don't match expected distribution. */
  SHAFFT_ERR_SIZE_OVERFLOW = 13,  /**< Size exceeds INT_MAX. */
  SHAFFT_ERR_NOT_IMPL = 14,       /**< Feature not yet implemented. */
  SHAFFT_ERR_INVALID_STATE = 15,  /**< Operation not valid in current plan state. */
  SHAFFT_ERR_INTERNAL = 16        /**< Uncategorized internal error. */
} shafft_status_t;

/**
 * @brief Error source domain for detailed diagnostics.
 * @ingroup c_api
 *
 * When a SHAFFT function returns an error, call shafftLastErrorSource()
 * to determine which subsystem caused the failure.
 */
// NOLINTNEXTLINE(modernize-use-using)
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
int shafftLastErrorStatus(void);

/**
 * @brief Get the error source domain from the last error on this thread.
 * @ingroup c_api
 * @return Error source (one of shafft_errsrc_t values).
 */
int shafftLastErrorSource(void);

/**
 * @brief Get the raw domain-specific error code from the last error.
 * @ingroup c_api
 *
 * The meaning of this code depends on shafftLastErrorSource():
 * - SHAFFT_ERRSRC_MPI: MPI error code (use MPI_Error_string)
 * - SHAFFT_ERRSRC_HIP: hipError_t value
 * - SHAFFT_ERRSRC_HIPFFT: hipfftResult_t value
 * - SHAFFT_ERRSRC_FFTW: (not used, FFTW has no error codes)
 * - SHAFFT_ERRSRC_SYSTEM: errno or similar
 *
 * @return Raw error code from the underlying subsystem.
 */
int shafftLastErrorDomainCode(void);

/**
 * @brief Get a human-readable message for the last error.
 * @ingroup c_api
 * @param buf    Buffer to receive the null-terminated message.
 * @param buflen Size of the buffer in bytes.
 * @return Number of characters written (excluding null terminator).
 */
int shafftLastErrorMessage(char* buf, int buflen);

/**
 * @brief Clear the last error state on this thread.
 * @ingroup c_api
 *
 * @return void.
 */
void shafftClearLastError(void);

/**
 * @brief Get the name of an error source as a string.
 * @ingroup c_api
 * @param source Error source value from shafftLastErrorSource().
 * @return String name (e.g., "MPI", "HIP", "hipFFT", "FFTW").
 */
const char* shafftErrorSourceName(int source);

/**
 * @brief Decomposition strategy for automatic configuration.
 * @ingroup c_api
 */
// NOLINTNEXTLINE(modernize-use-using)
typedef enum {
  SHAFFT_MAXIMIZE_NDA, /**< Maximize number of distributed axes (better parallelism) */
  SHAFFT_MINIMIZE_NDA  /**< Minimize number of distributed axes (fewer communication phases) */
} shafft_decomposition_strategy_t;

/**
 * @brief Flags bitset for config object state.
 * @ingroup c_api
 */
// NOLINTNEXTLINE(modernize-use-using)
typedef enum {
  SHAFFT_CONFIG_CHANGED_COMM_DIMS = (1 << 0), /**< commDims was adjusted by resolve. */
  SHAFFT_CONFIG_CHANGED_NDA = (1 << 1),       /**< nda was adjusted by resolve. */
  SHAFFT_CONFIG_INACTIVE_RANKS = (1 << 2),    /**< Some ranks are inactive. */
  SHAFFT_CONFIG_RESOLVED = (1 << 3)           /**< Config has been resolved. */
} shafft_config_flags_t;

/* ---- N-D layout sub-struct ------------------------------------------------*/

/**
 * @brief N-D layout state (initial or output).
 * @ingroup c_api
 *
 * Contains per-axis arrays describing the local block for one layout state.
 * All pointer fields are config-owned and populated by resolve.
 */
typedef struct shafft_nd_layout_t {
  size_t* subsize;      /**< Local extents per axis (length = ndim). */
  size_t* offset;       /**< Global starting indices per axis (length = ndim). */
  size_t localElements; /**< Product of subsize (total local elements). */
  int hasLocalElements; /**< Boolean: localElements > 0. */
  int* contiguousAxes;  /**< Contiguous axis indices (length = nca). */
  int* distributedAxes; /**< Distributed axis indices (length = nda). */
} shafft_nd_layout_t;

/* ---- N-D configuration struct ---------------------------------------------*/

/**
 * @brief N-D configuration object (C POD struct).
 * @ingroup c_api
 *
 * The authoritative data store for all language APIs.
 * Caller-owned (stack/heap); internal arrays are config-owned.
 * Do NOT copy by value (owning pointers — double-free hazard).
 * Use shafftConfigNDInit/shafftConfigNDRelease to manage lifecycle.
 */
typedef struct shafft_nd_config_t {
  size_t structSize; /**< Must be sizeof(shafft_nd_config_t). */
  int apiVersion;    /**< API version (set by init). */

  /* invariant inputs (set before resolve) */
  int ndim;            /**< Number of tensor dimensions. */
  shafft_t precision;  /**< FFT type (SHAFFT_C2C or SHAFFT_Z2Z). */
  size_t* globalShape; /**< Global extents per axis (config-owned, length = ndim). */
  shafft_decomposition_strategy_t strategy; /**< Decomposition strategy. */
  shafft_transform_layout_t outputPolicy;   /**< Forward output-layout policy. */
  size_t memLimit;                          /**< Per-rank memory limit in bytes (0 = no limit). */

  /* hint fields (read/write; may be adjusted by resolve) */
  int* hintCommDims; /**< Process grid hint (config-owned, length = ndim). */
  int hintNda;       /**< Number of distributed axes hint (0 = auto). */

  /* computed decomposition (read-only after resolve) */
  int* commDims;        /**< Resolved process grid (config-owned, length = ndim). */
  int nda;              /**< Resolved number of distributed axes. */
  int commSize;         /**< Number of active ranks. */
  size_t allocElements; /**< Required buffer size in complex elements. */

  /* computed layouts */
  shafft_nd_layout_t initial; /**< Layout before execution. */
  shafft_nd_layout_t output;  /**< Post-forward layout (policy-dependent). */

  /* activity metadata */
  int isActive;   /**< Boolean: this rank participates. */
  int activeRank; /**< Rank within active subset; -1 if inactive. */
  int activeSize; /**< Number of active ranks; 0 before resolve. */

  /* communicator fields (owned by config; freed by Release) */
  MPI_Comm worldComm;  /**< Dup of communicator passed to init. */
  MPI_Comm activeComm; /**< Subcommunicator of active ranks; MPI_COMM_NULL if inactive. */

  /* topology metadata */
  int nodeId;           /**< Node identifier within the communicator. */
  int nodeCount;        /**< Total number of distinct nodes. */
  char* hostname;       /**< Config-owned hostname string. */
  size_t hostnameLen;   /**< Length of hostname (excluding null). */
  char* deviceName;     /**< Config-owned device name; NULL if not applicable. */
  size_t deviceNameLen; /**< Length of deviceName (excluding null). */

  int flags;       /**< Bitwise OR of shafft_config_flags_t values. */
  int status;      /**< Last resolve/init status code. */
  int reserved[8]; /**< Reserved for future use. */
} shafft_nd_config_t;

/* ---- 1-D layout sub-struct ------------------------------------------------*/

/**
 * @brief 1-D layout state (initial or output).
 * @ingroup c_api
 */
typedef struct shafft_1d_layout_t {
  size_t localSize;     /**< Elements on this rank. */
  size_t localStart;    /**< Offset in global array. */
  int hasLocalElements; /**< Boolean: localSize > 0. */
} shafft_1d_layout_t;

/* ---- 1-D configuration struct ---------------------------------------------*/

/**
 * @brief 1-D configuration object (C POD struct).
 * @ingroup c_api
 *
 * Simpler than N-D: no adjustable hints, no strategy/policy.
 * Do NOT copy by value (owning pointers — double-free hazard).
 */
typedef struct shafft_1d_config_t {
  size_t structSize; /**< Must be sizeof(shafft_1d_config_t). */
  int apiVersion;    /**< API version (set by init). */

  /* invariant inputs */
  size_t globalSize;  /**< Global FFT length. */
  shafft_t precision; /**< SHAFFT_C2C or SHAFFT_Z2Z. */

  /* computed layout per state (valid after resolve) */
  shafft_1d_layout_t initial; /**< Layout before execution. */
  shafft_1d_layout_t output;  /**< Post-forward layout. */
  size_t allocElements;       /**< Required buffer size in complex elements. */

  /* activity metadata */
  int isActive;   /**< Boolean. */
  int activeRank; /**< Rank within active subset; -1 if inactive. */
  int activeSize; /**< Number of active ranks; 0 before resolve. */

  /* communicator fields (owned by config; freed by Release) */
  MPI_Comm worldComm;  /**< Dup of communicator passed to init. */
  MPI_Comm activeComm; /**< Subcommunicator of active ranks; MPI_COMM_NULL if inactive. */

  /* topology metadata */
  int nodeId;           /**< Node identifier. */
  int nodeCount;        /**< Total number of distinct nodes. */
  char* hostname;       /**< Config-owned. */
  size_t hostnameLen;   /**< Length of hostname. */
  char* deviceName;     /**< Config-owned; NULL if not applicable. */
  size_t deviceNameLen; /**< Length of deviceName. */

  int flags;       /**< Bitwise OR of shafft_config_flags_t values. */
  int status;      /**< Last resolve/init status code. */
  int reserved[8]; /**< Reserved for future use. */
} shafft_1d_config_t;

/* ---- Config object functions ----------------------------------------------*/

/**
 * @brief Initialize and resolve an N-D configuration object.
 * @ingroup c_api
 *
 * Allocates internal arrays, copies inputs, resolves decomposition,
 * populates layouts and metadata, stores worldComm (dup of comm),
 * and creates activeComm via MPI_Comm_split.
 * The struct must be zero-initialized before this call.
 * Collective on @p comm.
 *
 * @param cfg          [in,out] Config struct (caller-owned, zero-initialized).
 * @param ndim         Number of tensor dimensions (>= 1).
 * @param globalShape  Global extents per axis (length = ndim).
 * @param precision    FFT type (SHAFFT_C2C or SHAFFT_Z2Z).
 * @param commDims     Process grid hint (length = ndim; NULL = auto).
 * @param hintNda      Number of distributed axes hint (0 = auto).
 * @param strategy     Decomposition strategy.
 * @param outputPolicy Forward output-layout policy (0 = REDISTRIBUTED default).
 * @param memLimit     Per-rank memory limit in bytes (0 = no limit).
 * @param comm         MPI communicator.
 * @return 0 on success, non-zero on error.
 */
int shafftConfigNDInit(shafft_nd_config_t* cfg,
                       int ndim,
                       const size_t* globalShape,
                       shafft_t precision,
                       const int* commDims,
                       int hintNda,
                       shafft_decomposition_strategy_t strategy,
                       shafft_transform_layout_t outputPolicy,
                       size_t memLimit,
                       MPI_Comm comm);

/**
 * @brief Release internal resources of an N-D config object.
 * @ingroup c_api
 *
 * Frees config-owned arrays/strings and owned communicators (worldComm,
 * activeComm), then zero-fills the struct.
 * Must be called before MPI_Finalize.
 * Does not free the struct itself. Idempotent; safe after failed init.
 *
 * @param cfg Config struct.
 */
void shafftConfigNDRelease(shafft_nd_config_t* cfg);

/**
 * @brief Re-resolve an N-D configuration.
 * @ingroup c_api
 *
 * Uses stored worldComm (no comm parameter). Frees old activeComm
 * and creates a new one. For advanced use after modifying struct fields.
 *
 * @param cfg  [in,out] Initialized config struct.
 * @return 0 on success, non-zero on error.
 */
int shafftConfigNDResolve(shafft_nd_config_t* cfg);

/**
 * @brief Initialize an N-D plan from a resolved config object.
 * @ingroup c_api
 *
 * Auto-resolves if SHAFFT_CONFIG_RESOLVED is not set.
 * Communicator is read from the config struct (worldComm).
 *
 * @param planPtr Plan from shafftNDCreate().
 * @param cfg     [in,out] Config struct (resolved or will be auto-resolved).
 * @return 0 on success, non-zero on error.
 */
int shafftNDInitFromConfig(void* planPtr, shafft_nd_config_t* cfg);

/**
 * @brief Initialize and resolve a 1-D configuration object.
 * @ingroup c_api
 *
 * Allocates internals, copies inputs, resolves layout, stores worldComm
 * (dup of comm), and creates activeComm.
 * Collective on @p comm.
 *
 * @param cfg        [in,out] Config struct (caller-owned, zero-initialized).
 * @param globalSize Global FFT length (> 0).
 * @param precision  FFT type (SHAFFT_C2C or SHAFFT_Z2Z).
 * @param comm       MPI communicator.
 * @return 0 on success, non-zero on error.
 */
int shafftConfig1DInit(shafft_1d_config_t* cfg,
                       size_t globalSize,
                       shafft_t precision,
                       MPI_Comm comm);

/**
 * @brief Release internal resources of a 1-D config object.
 * @ingroup c_api
 *
 * Frees config-owned strings and owned communicators (worldComm,
 * activeComm), then zero-fills the struct.
 * Must be called before MPI_Finalize.
 *
 * @param cfg Config struct.
 */
void shafftConfig1DRelease(shafft_1d_config_t* cfg);

/**
 * @brief Re-resolve a 1-D configuration.
 * @ingroup c_api
 *
 * Uses stored worldComm (no comm parameter). Frees old activeComm
 * and creates a new one.
 *
 * @param cfg  [in,out] Initialized config struct.
 * @return 0 on success, non-zero on error.
 */
int shafftConfig1DResolve(shafft_1d_config_t* cfg);

/**
 * @brief Initialize a 1-D plan from a resolved config object.
 * @ingroup c_api
 *
 * Auto-resolves if SHAFFT_CONFIG_RESOLVED is not set.
 * Communicator is read from the config struct (worldComm).
 *
 * @param planPtr Plan from shafft1DCreate().
 * @param cfg     [in,out] Config struct (resolved or will be auto-resolved).
 * @return 0 on success, non-zero on error.
 */
int shafft1DInitFromConfig(void* planPtr, shafft_1d_config_t* cfg);

/**
 * @brief Get a duplicated communicator from a plan.
 * @ingroup c_api
 *
 * Returns MPI_COMM_NULL on inactive ranks. The caller is responsible
 * for MPI_Comm_free() on the returned non-null communicator.
 * Valid only after planning stage (shafftPlan).
 *
 * @param planPtr Plan pointer (N-D or 1D).
 * @param outComm [out] Receives duplicated communicator.
 * @return 0 on success, non-zero on error.
 */
int shafftGetCommunicator(void* planPtr, MPI_Comm* outComm);

/**
 * @brief Compute local layout and process grid for N-D distributed FFT.
 * @ingroup c_api
 *
 * Attempts configuration using the fallback chain: commDims -> nda -> strategy.
 *
 * @param ndim       Number of tensor dimensions.
 * @param size       Global extents per axis (length = ndim).
 * @param precision  FFT type (SHAFFT_C2C or SHAFFT_Z2Z).
 * @param commDims   [in,out] Process grid preference (0 = no preference); updated on output.
 * @param nda        [in,out] Distributed axes preference (0 = no preference); updated on output.
 * @param subsize    [out] Local extents per axis (length = ndim).
 * @param offset     [out] Global starting indices per axis (length = ndim).
 * @param commSize   [out] Number of active ranks.
 * @param strategy   Fallback strategy (SHAFFT_MAXIMIZE_NDA or SHAFFT_MINIMIZE_NDA).
 * @param memLimit   Per-rank memory limit in bytes (0 = no limit).
 * @param cComm     MPI communicator.
 * @return 0 on success, non-zero on error.
 */
int shafftConfigurationND(int ndim,
                          int* size,
                          shafft_t precision,
                          int* commDims,
                          int* nda,
                          size_t* subsize,
                          size_t* offset,
                          int* commSize,
                          shafft_decomposition_strategy_t strategy,
                          size_t memLimit,
                          MPI_Comm cComm);

/**
 * @brief Allocate uninitialized N-D plan object.
 * @ingroup c_api
 *
 * @param outPlan [out] Receives plan pointer (NULL on failure).
 * @return 0 on success, non-zero on error.
 */
int shafftNDCreate(void** outPlan);

/**
 * @brief Initialize N-D plan from process grid.
 * @ingroup c_api
 *
 * @param planPtr       Plan from shafftNDCreate().
 * @param ndim          Number of dimensions.
 * @param commDims      Process grid dimensions (length = ndim).
 * @param dimensions    Global tensor extents (length = ndim).
 * @param precision     FFT type (SHAFFT_C2C or SHAFFT_Z2Z).
 * @param cComm         MPI communicator.
 * @param output_policy Forward output-layout policy.
 * @return 0 on success, non-zero on error.
 */
int shafftNDInit(void* planPtr,
                 int ndim,
                 int commDims[],
                 int dimensions[],
                 shafft_t precision,
                 MPI_Comm cComm,
                 shafft_transform_layout_t output_policy);

/**
 * @brief Create backend FFT plans.
 * @ingroup c_api
 *
 * Must be called after init completes.
 * Works on N-D and 1D plans. Calling plan() more than once is an error.
 *
 * @param planPtr Plan pointer (N-D or 1D).
 * @return 0 on success, non-zero on error.
 */
int shafftPlan(void* planPtr);

/**
 * @brief Release plan resources.
 * @ingroup c_api
 *
 * Works on N-D and 1D plans. Does not free user-provided buffers.
 *
 * @param planPtr [in,out] Pointer to plan; set to NULL on return.
 * @return 0 on success, non-zero on error.
 */
int shafftDestroy(void** planPtr);

/**
 * @brief Query local tensor layout.
 * @ingroup c_api
 *
 * @param planPtr Plan pointer (N-D or 1D).
 * @param subsize  [out] Local extents per axis.
 * @param offset   [out] Global starting indices per axis.
 * @param layout   Which layout to query.
 * @return 0 on success, non-zero on error.
 */
int shafftGetLayout(void* planPtr, size_t* subsize, size_t* offset, shafft_tensor_layout_t layout);

/**
 * @brief Query contiguous and distributed axis indices.
 * @ingroup c_api
 *
 * @param planPtr Plan pointer (N-D or 1D).
 * @param ca       [out] Contiguous axis indices (innermost to outermost stride).
 * @param da       [out] Distributed axis indices (innermost to outermost stride).
 * @param layout   Which layout to query.
 * @return 0 on success, non-zero on error.
 */
int shafftGetAxes(void* planPtr, int* ca, int* da, shafft_tensor_layout_t layout);

/**
 * @brief Get required buffer size in complex elements.
 * @ingroup c_api
 *
 * @param plan           Plan pointer (N-D or 1D).
 * @param localAllocSize [out] Required element count.
 * @return 0 on success, non-zero on error.
 */
int shafftGetAllocSize(void* plan, size_t* localAllocSize);

/**
 * @brief Get global FFT size (product of all dimensions).
 * @ingroup c_api
 *
 * @param plan        Plan pointer (N-D or 1D).
 * @param globalSize [out] Total element count.
 * @return 0 on success, non-zero on error.
 */
int shafftGetGlobalSize(void* plan, size_t* globalSize);

/**
 * @brief Check if plan is configured (init succeeded).
 * @ingroup c_api
 *
 * @param plan          Plan pointer (N-D or 1D).
 * @param configured [out] 1 if configured, 0 otherwise.
 * @return 0 on success, non-zero on error.
 */
int shafftIsConfigured(void* plan, int* configured);

/**
 * @brief Check if this rank participates in the plan.
 * @ingroup c_api
 *
 * @param plan      Plan pointer (N-D or 1D).
 * @param active [out] 1 if active, 0 otherwise.
 * @return 0 on success, non-zero on error.
 */
int shafftIsActive(void* plan, int* active);

/**
 * @brief Attach data and work buffers to a plan.
 * @ingroup c_api
 *
 * Buffers must remain valid until plan destruction or new buffers are set.
 *
 * @param planPtr Plan pointer (N-D or 1D).
 * @param data     Main data buffer.
 * @param work     Work/scratch buffer.
 * @return 0 on success, non-zero on error.
 */
int shafftSetBuffers(void* planPtr, void* data, void* work);

/**
 * @brief Retrieve current buffer pointers.
 * @ingroup c_api
 *
 * Buffers may be swapped during execution; call after shafftExecute()
 * to locate the most recent execution result.
 *
 * @param planPtr Plan pointer (N-D or 1D).
 * @param data     [out] Current data buffer.
 * @param work     [out] Current work buffer.
 * @return 0 on success, non-zero on error.
 */
int shafftGetBuffers(void* planPtr, void** data, void** work);

/**
 * @brief Execute FFT transform.
 * @ingroup c_api
 *
 * @param planPtr  Plan pointer (N-D or 1D).
 * @param direction SHAFFT_FORWARD or SHAFFT_BACKWARD.
 * @return 0 on success, non-zero on error.
 */
int shafftExecute(void* planPtr, shafft_direction_t direction);

/**
 * @brief Apply normalization to data buffer.
 * @ingroup c_api
 *
 * @param planPtr Plan pointer (N-D or 1D).
 * @return 0 on success, non-zero on error.
 */
int shafftNormalize(void* planPtr);

/**
 * @brief Compute local layout for 1D distributed FFT.
 * @ingroup c_api
 *
 * @param N              Global FFT size.
 * @param localN         [out] Elements for this rank.
 * @param localStart     [out] This rank's offset in global array.
 * @param localAllocSize [out] Required buffer size.
 * @param precision      FFT type (SHAFFT_C2C or SHAFFT_Z2Z).
 * @param cComm         MPI communicator.
 * @return 0 on success, non-zero on error.
 */
int shafftConfiguration1D(size_t N,
                          size_t* localN,
                          size_t* localStart,
                          size_t* localAllocSize,
                          shafft_t precision,
                          MPI_Comm cComm);

/**
 * @brief Allocate uninitialized 1D plan object.
 * @ingroup c_api
 *
 * @param outPlan [out] Receives plan pointer.
 * @return 0 on success, non-zero on error.
 */
int shafft1DCreate(void** outPlan);

/**
 * @brief Initialize 1D distributed FFT plan.
 * @ingroup c_api
 *
 * @param planPtr    Plan from shafft1DCreate().
 * @param N          Global FFT size.
 * @param localN     Elements for this rank.
 * @param localStart This rank's offset.
 * @param precision  FFT type (SHAFFT_C2C or SHAFFT_Z2Z).
 * @param cComm      MPI communicator.
 * @return 0 on success, non-zero on error.
 */
int shafft1DInit(
    void* planPtr, size_t N, size_t localN, size_t localStart, shafft_t precision, MPI_Comm cComm);

/**
 * @brief Allocate single-precision complex buffer.
 * @ingroup c_api
 *
 * Uses device memory on GPU, host memory on CPU. Free with shafftFreeBufferF().
 *
 * @param count Number of complex elements.
 * @param buf   [out] Allocated buffer.
 * @return 0 on success, non-zero on error.
 */
int shafftAllocBufferF(size_t count, void** buf);

/**
 * @brief Allocate double-precision complex buffer.
 * @ingroup c_api
 *
 * Uses device memory on GPU, host memory on CPU. Free with shafftFreeBufferD().
 *
 * @param count Number of complex elements.
 * @param buf   [out] Allocated buffer.
 * @return 0 on success, non-zero on error.
 */
int shafftAllocBufferD(size_t count, void** buf);

/**
 * @brief Free single-precision buffer from shafftAllocBufferF().
 * @ingroup c_api
 * @param buf Buffer to free (NULL safe).
 * @return 0 on success, non-zero on error.
 */
int shafftFreeBufferF(void* buf);

/**
 * @brief Free double-precision buffer from shafftAllocBufferD().
 * @ingroup c_api
 * @param buf Buffer to free (NULL safe).
 * @return 0 on success, non-zero on error.
 */
int shafftFreeBufferD(void* buf);

/**
 * @brief Copy single-precision data from host to SHAFFT buffer.
 * @ingroup c_api
 *
 * @param dst   Destination buffer.
 * @param src   Source host memory.
 * @param count Number of complex elements.
 * @return 0 on success, non-zero on error.
 */
int shafftCopyToBufferF(void* dst, const void* src, size_t count);

/**
 * @brief Copy double-precision data from host to SHAFFT buffer.
 * @ingroup c_api
 *
 * @param dst   Destination buffer.
 * @param src   Source host memory.
 * @param count Number of complex elements.
 * @return 0 on success, non-zero on error.
 */
int shafftCopyToBufferD(void* dst, const void* src, size_t count);

/**
 * @brief Copy single-precision data from SHAFFT buffer to host.
 * @ingroup c_api
 *
 * @param dst   Destination host memory.
 * @param src   Source buffer.
 * @param count Number of complex elements.
 * @return 0 on success, non-zero on error.
 */
int shafftCopyFromBufferF(void* dst, const void* src, size_t count);

/**
 * @brief Copy double-precision data from SHAFFT buffer to host.
 * @ingroup c_api
 *
 * @param dst   Destination host memory.
 * @param src   Source buffer.
 * @param count Number of complex elements.
 * @return 0 on success, non-zero on error.
 */
int shafftCopyFromBufferD(void* dst, const void* src, size_t count);

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
 * @brief Get the library version as a string.
 * @ingroup c_api
 * @return Version string.
 */
const char* shafftGetVersionString(void);

/**
 * @brief Finalize library and release backend resources.
 * @ingroup c_api
 *
 * Call after all plans are destroyed. Must be called before MPI_Finalize()
 * for FFTW backend. Safe to call multiple times.
 *
 * @return 0 on success, non-zero on error.
 */
int shafftFinalize(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif // SHAFFT_C_H
