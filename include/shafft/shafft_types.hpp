/** @file shafft_types.hpp
 *  @brief Core types and execution helpers for SHAFFT.
 *  @ingroup cpp_api
 */

#ifndef SHAFFT_TYPES_H
#define SHAFFT_TYPES_H

#include <shafft/shafft_config.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <mpi.h>
#include <vector>

/// @brief Opaque handle for per-axis device FFT subplans (backend-specific).
struct fftHandle;

namespace shafft {

/**
 * @brief FFT backend identifier.
 * @ingroup cpp_api
 */
enum class Backend { HIPFFT, FFTW };

/**
 * @brief Get the compile-time backend.
 * @ingroup cpp_api
 */
constexpr Backend backend() noexcept {
#if SHAFFT_BACKEND_HIPFFT
  return Backend::HIPFFT;
#else
  return Backend::FFTW;
#endif
}

/**
 * @brief Check if the current backend is GPU-based.
 * @ingroup cpp_api
 */
constexpr bool isGPU() noexcept {
  return backend() == Backend::HIPFFT;
}

/**
 * @brief Get the backend name as a string.
 * @ingroup cpp_api
 */
constexpr const char* backendName() noexcept {
#if SHAFFT_BACKEND_HIPFFT
  return "hipFFT";
#else
  return "FFTW";
#endif
}

/// @brief Portable single-precision complex type (maps to backend type internally).
/// @ingroup cpp_api
using complexf = std::complex<float>;
/// @brief Portable double-precision complex type (maps to backend type internally).
/// @ingroup cpp_api
using complexd = std::complex<double>;

//------------------------------------------------------------------------------
// Memory spaces and typed pointer wrappers
//------------------------------------------------------------------------------

/**
 * @brief Memory space for typed pointers.
 * @ingroup cpp_api
 */
enum class MemorySpace { Host, Device };

/**
 * @brief Typed pointer wrapper with memory space annotation.
 * @ingroup cpp_api
 * @tparam T Element type.
 * @tparam Space Memory space (Host or Device).
 */
template <typename T, MemorySpace Space>
struct Ptr {
  T* p = nullptr;                                                   ///< Raw pointer.
  T* get() const noexcept { return p; }                             ///< Get the raw pointer.
  explicit operator bool() const noexcept { return p != nullptr; }  ///< Check if non-null.
};

/// @brief Host pointer to single-precision complex.
/// @ingroup cpp_api
using HostPtr = Ptr<complexf, MemorySpace::Host>;
/// @brief Device pointer to single-precision complex.
/// @ingroup cpp_api
using DevicePtr = Ptr<complexf, MemorySpace::Device>;
/// @brief Host pointer to double-precision complex.
/// @ingroup cpp_api
using HostPtrZ = Ptr<complexd, MemorySpace::Host>;
/// @brief Device pointer to double-precision complex.
/// @ingroup cpp_api
using DevicePtrZ = Ptr<complexd, MemorySpace::Device>;

#if SHAFFT_BACKEND_HIPFFT
/// @brief Native pointer type for the current backend (single precision).
using NativePtr = DevicePtr;
/// @brief Native pointer type for the current backend (double precision).
using NativePtrZ = DevicePtrZ;
#else
/// @brief Native pointer type for the current backend (single precision).
using NativePtr = HostPtr;
/// @brief Native pointer type for the current backend (double precision).
using NativePtrZ = HostPtrZ;
#endif

/// @brief Traits to get the backend complex type from a scalar type.
template <typename T>
struct backend_complex;
/// @brief Specialization for float -> complexf.
template <>
struct backend_complex<float> {
  using type = complexf;
};
/// @brief Specialization for double -> complexd.
template <>
struct backend_complex<double> {
  using type = complexd;
};

/**
 * @brief Status and error codes.
 * @ingroup cpp_api
 */
enum class Status : int {
  SHAFFT_SUCCESS,              ///< Operation succeeded.
  SHAFFT_ERR_NULLPTR,          ///< A required pointer argument was null.
  SHAFFT_ERR_INVALID_COMM,     ///< Invalid or unsupported MPI communicator.
  SHAFFT_ERR_NO_BUFFER,        ///< Required data/work buffer was not set.
  SHAFFT_ERR_PLAN_NOT_INIT,    ///< Plan or subplan not initialized.
  SHAFFT_ERR_INVALID_DIM,      ///< Invalid dimension/rank/size.
  SHAFFT_ERR_DIM_MISMATCH,     ///< Dimension mismatch between inputs.
  SHAFFT_ERR_INVALID_DECOMP,   ///< Invalid or unsupported slab decomposition.
  SHAFFT_ERR_INVALID_FFTTYPE,  ///< Unsupported FFTType.
  SHAFFT_ERR_ALLOC,            ///< Memory allocation failure.
  SHAFFT_ERR_BACKEND,          ///< Local FFT backend failure.
  SHAFFT_ERR_MPI,              ///< MPI failure.
  SHAFFT_ERR_INTERNAL          ///< Uncategorized internal error.
};

/**
 * @brief FFT transform type.
 * @ingroup cpp_api
 * - C2C: single-precision complex-to-complex.
 * - Z2Z: double-precision complex-to-complex.
 */
enum class FFTType { C2C, Z2Z };

/**
 * @brief Transform direction.
 * @ingroup cpp_api
 */
enum class FFTDirection { FORWARD, BACKWARD };

/**
 * @brief Tensor layout.
 * @ingroup cpp_api
 * - CURRENT: current layout (may be initial or transformed).
 * - INITIAL: initial layout (as provided to plan creation).
 * - TRANSFORMED: layout after the FFT has been applied (post forward or backward FFT).
 */
enum class TensorLayout { CURRENT, INITIAL, TRANSFORMED };

/**
 * @brief Get the size of an FFT element for a given type.
 * @ingroup cpp_api
 */
constexpr std::size_t sizeof_fft_element(FFTType t) noexcept {
  return (t == FFTType::C2C) ? sizeof(complexf) : sizeof(complexd);
}

/**
 * @brief Compute the product of the first @p ndim entries of @p array.
 * @ingroup cpp_api
 * @tparam T Input element type.
 * @tparam U Accumulator/return type.
 * @param array Pointer to at least @p ndim elements.
 * @param ndim Number of dimensions/elements to multiply.
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

/**
 * @brief Tensor slab decomposition and axes redistribution.
 * @ingroup cpp_api
 *
 * Manages MPI communicator layouts, local subarray metadata, buffer attachment and
 * staged exchanges required to execute the multidimensional FFT.
 *
 * - **Contiguous axes (`ca`)** are the locally contiguous, non-distributed axes
 *   for a given stage/layout; order is innermost to outermost stride.
 */
class Slab {
 public:
  /**
   * @brief Construct with explicit Cartesian process grid.
   * @param ndim Global rank.
   * @param size Global extents (length = @p ndim).
   * @param COMM_DIMS Cartesian process-grid dimensions (length = grid rank).
   * @param MPI_sendtype MPI datatype.
   * @param comm MPI communicator.
   * @param elem_size Bytes per element (only used when SHAFFT_GPU_AWARE_MPI=0).
   */
  Slab(int ndim, int size[], int COMM_DIMS[], MPI_Datatype MPI_sendtype, MPI_Comm comm,
       size_t elem_size = 0);

  /// @brief Destructor; releases internal MPI/datatype resources.
  ~Slab();

  /**
   * @brief Get the @p ith stage configuration for this rank.
   * @param subsize [out] Local extents per axis.
   * @param offset  [out] Global start indices per axis.
   * @param ca      [out] Contiguous (non-distributed) axis indices, innermost to outermost.
   * @param ith     Stage index (0-based).
   * @return 0 on success, non-zero on error.
   */
  int get_ith_config(int* subsize, int* offset, int* ca, int ith);

  /**
   * @brief Get the @p ith stage layout for this rank.
   * @param subsize [out] Local extents per axis.
   * @param offset  [out] Global start indices per axis.
   * @param ith     Stage index (0-based).
   * @return 0 on success, non-zero on error.
   */
  int get_ith_layout(int* subsize, int* offset, int ith);

  /**
   * @brief Get the @p ith stage axes for this rank.
   * @param ca      [out] Contiguous (non-distributed) axis indices, innermost to outermost.
   * @param da      [out] Distributed axis indices, innermost to outermost.
   * @param ith     Stage index (0-based).
   * @return 0 on success, non-zero on error.
   */
  int get_ith_axes(int* ca, int* da, int ith);

  /**
   * @brief Total buffer size required (in elements) across data/work.
   * @return Required elements for attached buffers.
   */
  size_t alloc_size();

  /**
   * @brief Attach data and work buffers.
   * @param Adata Data buffer pointer.
   * @param Bdata Work/scratch buffer pointer.
   */
  void set_buffers(void* Adata, void* Bdata);

  /**
   * @brief Retrieve currently attached data and work buffer pointers.
   * @param Adata [out] Receives data pointer.
   * @param Bdata [out] Receives work pointer.
   */
  void get_buffers(void** Adata, void** Bdata);

  /// @brief Swap the internal roles of data and work buffers.
  void swap_buffers();

  /// @brief Execute an axes redistribution in the forward (FFTDirection::FORWARD) pipeline.
  /// @return 0 on success, non-zero on error.
  int forward();

  /// @brief Execute an axes redistribution in the backward (FFTDirection::BACKWARD) pipeline.
  /// @return 0 on success, non-zero on error.
  int backward();

  /// @brief Current exchange stage index (0-based).
  int es_index();

  /// @brief Number of exchange stages in the forward/backward pipelines.
  int nes();

  /// @brief Get full list of exchange stage indices into @p es (length = nes()).
  void get_es(int* es);

  /// @brief Copy global extents into @p size (length = ndim()).
  void get_size(int* size);

  /// @brief Copy current local extents into @p subsize (length = ndim()).
  void get_subsize(int* subsize);

  /// @brief Copy current global offsets into @p offset (length = ndim()).
  void get_offset(int* offset);

  /// @brief Get the current data buffer pointer.
  void* data();

  /// @brief Get the current work buffer pointer.
  void* work();

  /// @brief Global rank (number of axes).
  int ndim();

  /// @brief Number of contiguous (non-distributed) axes in the current stage.
  int nca();

  /// @brief Get contiguous axis indices into @p ca (length = nca()).
  void get_ca(int* ca);

  /// @brief Number of distributed axes (NDA rank).
  int nda();

  /// @brief Get distributed axis indices into @p da (length = nda()).
  void get_da(int* da);

  /// @brief Check if this rank is active (has work to do).
  /// @return true if this rank participates in the FFT, false if excluded.
  bool is_active() const noexcept;

  /**
   * @brief Convert a flat @p index into multi-index coordinates.
   * @param index   Linear index.
   * @param ndim    Rank (number of axes).
   * @param size    Extents per axis (length = @p ndim).
   * @param offset  Offsets per axis (length = @p ndim).
   * @param indices [out] Multi-index (length = @p ndim).
   */
  void get_indices(int index, int ndim, int* size, int* offset, int* indices);

  /**
   * @brief Convert a multi-index into a flat (linear) index.
   * @param indices Multi-index (length = @p ndim).
   * @param ndim    Rank (number of axes).
   * @param size    Extents per axis (length = @p ndim).
   * @return Linear index.
   */
  int get_index(int* indices, int ndim, int* size);

 private:
  int _taA;
  int _taB;
  int _nes;
  int _ndim;
  int* _size;
  int* _subsize;
  int* _offset;
  int* _subsizes;
  int* _offsets;
  int _nca;
  int _nda;
  int* _caA;
  int* _caB;
  int* _cas;
  int* _das;
  int* _axesA;
  int* _axesB;
  int* _daA;
  int* _daB;
  int _exchange_index = 0;
  int* _es;
  int _es_index = 0;
  int* _subsizeA;
  int* _subsizeB;
  int _exchange_direction;
  MPI_Datatype* _subarrayA;
  MPI_Datatype* _subarrayB;
  void* _Adata = nullptr;
  void* _Bdata = nullptr;
  MPI_Comm* _worldcomm;
  MPI_Comm* _comms;
  MPI_Comm _comm;
  int _max_comm_size;
  MPI_Datatype* _subarrays;

#if SHAFFT_BACKEND_HIPFFT && !SHAFFT_GPU_AWARE_MPI
  // Host-staging buffers for non-GPU-aware MPI (compile-time selected)
  size_t _elem_size = 0;   ///< Bytes per element (for host staging)
  void* _hostA = nullptr;  ///< Host staging buffer A (allocated lazily)
  void* _hostB = nullptr;  ///< Host staging buffer B (allocated lazily)
#endif

  /// @brief Prepare metadata for the forward axes redistribution.
  void prepare_forward_exchange();

  /// @brief Prepare metadata for the backward axes redistribution.
  void prepare_backward_exchange();

  /// @brief Execute an axes redistribution in the prepared direction.
  void _exchange();
};

/**
 * @brief Internal SHAFFT plan data structure.
 * @ingroup cpp_api
 *
 * Holds the decomposed tensor in a Slab instance, per-axis device subplans and
 * normalization parameters.
 *
 * @note Users should prefer the RAII shafft::Plan class over using PlanData directly.
 */
struct PlanData {
  Slab* slab = nullptr;              ///< Handles tensor decomposition and axes redistribution.
  int nsubplans = 0;                 ///< Number of backend subplans.
  fftHandle* subplans = nullptr;     ///< Array of backend subplans.
  FFTType fft_type;                  ///< FFT element/type (C2C or Z2Z).
  long double norm_denominator = 1;  ///< Normalization denominator (scale = (1/den)^exp).
  int norm_exponent = 0;             ///< Normalization exponent.
  TensorLayout current_layout = TensorLayout::INITIAL;  ///< Current tensor layout.
};

namespace err {

[[nodiscard]] constexpr const char* statusToString(Status s) noexcept {
  switch (s) {
    case Status::SHAFFT_SUCCESS:
      return "SUCCESS";
    case Status::SHAFFT_ERR_NULLPTR:
      return "NULLPTR";
    case Status::SHAFFT_ERR_INVALID_COMM:
      return "INVALID_COMM";
    case Status::SHAFFT_ERR_NO_BUFFER:
      return "NO_BUFFER";
    case Status::SHAFFT_ERR_PLAN_NOT_INIT:
      return "PLAN_NOT_INIT";
    case Status::SHAFFT_ERR_INVALID_DIM:
      return "INVALID_DIM";
    case Status::SHAFFT_ERR_DIM_MISMATCH:
      return "DIM_MISMATCH";
    case Status::SHAFFT_ERR_INVALID_DECOMP:
      return "INVALID_DECOMP";
    case Status::SHAFFT_ERR_INVALID_FFTTYPE:
      return "INVALID_FFTTYPE";
    case Status::SHAFFT_ERR_ALLOC:
      return "ALLOC";
    case Status::SHAFFT_ERR_BACKEND:
      return "BACKEND";
    case Status::SHAFFT_ERR_MPI:
      return "MPI";
    case Status::SHAFFT_ERR_INTERNAL:
      return "INTERNAL";
  }
  // No default: lets -Wswitch-enum warn if you add a new Status and forget to map it.
  return "UNKNOWN";
}

[[nodiscard]] constexpr const char* statusToString(int code) noexcept {
  return statusToString(static_cast<Status>(code));
}

}  // namespace err

}  // namespace shafft

#endif  // SHAFFT_TYPES_H
