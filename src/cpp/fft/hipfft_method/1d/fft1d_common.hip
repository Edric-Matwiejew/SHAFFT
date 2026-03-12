// fft1d_common.hip
// Shared infrastructure for distributed 1D FFT implementations.
// Contains common defines, buffer structs, MPI helpers, and kernel launch utilities.

#ifndef SHAFFT_FFT1D_COMMON_HIP
#define SHAFFT_FFT1D_COMMON_HIP

#include "../common/hipcheck.hpp"

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>
#include <mpi.h>

#include <cstddef>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// SHAFFT_GPU_AWARE_MPI provided by shafft_options compile definition

namespace shafft {

//==============================================================================
// Kernel launch configuration
//==============================================================================

/// Block size for 1D FFT kernels
static constexpr int kFFT1DBlockSize = 256;

/// Compute grid dimensions for n elements
inline dim3 fft1dGridSize(size_t n) {
  return dim3((n + kFFT1DBlockSize - 1) / kFFT1DBlockSize);
}

/// Get block dimensions
inline dim3 fft1dBlockSize() {
  return dim3(kFFT1DBlockSize);
}

//==============================================================================
// Buffer context for distributed FFT execution
//==============================================================================

/// Holds all buffers needed for distributed FFT execution.
/// Abstracts away GPU-aware MPI differences.
template <typename ComplexT>
struct FFT1DBuffers {
  ComplexT* dTemp; // Device: intermediate buffer
  ComplexT* dBuf1; // Device: working buffer 1
  ComplexT* dBuf2; // Device: working buffer 2
  ComplexT* hSend; // Host: send buffer (nullptr if GPU-aware MPI)
  ComplexT* hRecv; // Host: receive buffer (nullptr if GPU-aware MPI)
};

//==============================================================================
// MPI communication helpers with GPU-aware MPI abstraction
//==============================================================================

/// Perform MPI_Alltoall with automatic host staging if GPU-aware MPI is disabled.
/// @param dSrc      Device source buffer
/// @param dDst      Device destination buffer
/// @param hSend     Host send staging buffer (ignored if GPU-aware MPI)
/// @param hRecv     Host receive staging buffer (ignored if GPU-aware MPI)
/// @param count     Number of elements per rank (for MPI count parameter)
/// @param mpiType   MPI datatype (MPI_FLOAT or MPI_DOUBLE for complex components)
/// @param localSize Total elements in local buffer (for memcpy size)
/// @param comm      MPI communicator
template <typename ComplexT>
inline void alltoallWithStaging(ComplexT* dSrc,
                                ComplexT* dDst,
                                ComplexT* hSend,
                                ComplexT* hRecv,
                                int count,
                                MPI_Datatype mpiType,
                                size_t localSize,
                                MPI_Comm comm) {
#if SHAFFT_GPU_AWARE_MPI
  (void)hSend;
  (void)hRecv;
  (void)localSize;
  // Ensure all prior kernels on the default stream have completed before
  // MPI reads directly from device memory.
  hipCheck(hipStreamSynchronize(nullptr));
  MPI_Alltoall(dSrc, count, mpiType, dDst, count, mpiType, comm);
#else
  // hipMemcpy on the null stream is synchronous and acts as an implicit
  // device-wide barrier, so no explicit sync is needed here.
  hipCheck(hipMemcpy(hSend, dSrc, localSize * sizeof(ComplexT), hipMemcpyDeviceToHost));
  MPI_Alltoall(hSend, count, mpiType, hRecv, count, mpiType, comm);
  hipCheck(hipMemcpy(dDst, hRecv, localSize * sizeof(ComplexT), hipMemcpyHostToDevice));
#endif
}

/// Same as alltoallWithStaging but with swapped host buffer roles.
/// Used for alternating communication patterns to avoid extra copies.
template <typename ComplexT>
inline void alltoallWithStagingSwapped(ComplexT* dSrc,
                                       ComplexT* dDst,
                                       ComplexT* hSend,
                                       ComplexT* hRecv,
                                       int count,
                                       MPI_Datatype mpiType,
                                       size_t localSize,
                                       MPI_Comm comm) {
#if SHAFFT_GPU_AWARE_MPI
  (void)hSend;
  (void)hRecv;
  (void)localSize;
  // Ensure all prior kernels on the default stream have completed before
  // MPI reads directly from device memory.
  hipCheck(hipStreamSynchronize(nullptr));
  MPI_Alltoall(dSrc, count, mpiType, dDst, count, mpiType, comm);
#else
  // hipMemcpy on the null stream is synchronous and acts as an implicit
  // device-wide barrier, so no explicit sync is needed here.
  hipCheck(hipMemcpy(hRecv, dSrc, localSize * sizeof(ComplexT), hipMemcpyDeviceToHost));
  MPI_Alltoall(hRecv, count, mpiType, hSend, count, mpiType, comm);
  hipCheck(hipMemcpy(dDst, hSend, localSize * sizeof(ComplexT), hipMemcpyHostToDevice));
#endif
}

} // namespace shafft

#endif // SHAFFT_FFT1D_COMMON_HIP
