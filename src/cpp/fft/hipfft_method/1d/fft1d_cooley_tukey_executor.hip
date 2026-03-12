// fft1d_cooley_tukey_executor.hip
// Path A: Cooley-Tukey PxP distributed 1D FFT implementation.
// Contains GPU kernels and the executor function for the 10-step algorithm.
//
// Algorithm: Two-phase PxP decomposition
//   Phase 1: P-point FFTs on each chunk, with twiddle factors
//   Phase 2: L-point FFT on redistributed data
//   Communication: 3 MPI_Alltoall operations
//
// Requires N' % P^2 == 0 where N' is the (possibly padded) FFT size.

#ifndef SHAFFT_FFT1D_COOLEY_TUKEY_EXECUTOR_HIP
#define SHAFFT_FFT1D_COOLEY_TUKEY_EXECUTOR_HIP

#include "fft1d_common.hip"

#include <cmath>
#include <type_traits>

namespace shafft {

//==============================================================================
// GPU Kernels for Cooley-Tukey distributed FFT
//==============================================================================

/// Transpose Pxchunk -> chunkxP (scatter after first Alltoall)
template <typename ComplexT>
__global__ void fft1dScatterStep1Kernel(const ComplexT* __restrict__ in,
                                        ComplexT* __restrict__ out,
                                        size_t chunk,
                                        int numRanks,
                                        size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    size_t dst = (idx % chunk) * numRanks + (idx / chunk);
    out[dst] = in[idx];
  }
}

/// Transpose chunkxP -> Pxchunk (scatter before second Alltoall)
template <typename ComplexT>
__global__ void fft1dScatterStep4Kernel(const ComplexT* __restrict__ in,
                                        ComplexT* __restrict__ out,
                                        size_t chunk,
                                        int numRanks,
                                        size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    size_t dst = (idx % numRanks) * chunk + (idx / numRanks);
    out[dst] = in[idx];
  }
}

/// Twiddle multiply: w = exp(sign * 2pii * col * k1 / N)
/// Applied between the two FFT phases.
/// Uses sincos in double precision for accuracy, then casts to RealT.
template <typename ComplexT, typename RealT>
__global__ void fft1dTwiddleKernel(ComplexT* __restrict__ data,
                                   size_t chunk,
                                   int numRanks,
                                   int rank,
                                   size_t globalN,
                                   RealT sign,
                                   size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    size_t i = idx / numRanks;
    int k1 = idx % numRanks;
    size_t col = rank * chunk + i;

    // Always compute in double precision for accuracy
    double angle = static_cast<double>(sign) * (2.0 * M_PI) * static_cast<double>(col) *
                   static_cast<double>(k1) / static_cast<double>(globalN);

    double sinValD, cosValD;
    sincos(angle, &sinValD, &cosValD);
    RealT sinVal = static_cast<RealT>(sinValD);
    RealT cosVal = static_cast<RealT>(cosValD);

    ComplexT a = data[idx];
    data[idx].x = a.x * cosVal - a.y * sinVal;
    data[idx].y = a.x * sinVal + a.y * cosVal;
  }
}

/// Pack for redistribution: k = rank + k2 * numRanks, dest = k / localSize
template <typename ComplexT>
__global__ void fft1dPackKernel(const ComplexT* __restrict__ in,
                                ComplexT* __restrict__ out,
                                size_t chunk,
                                int rank,
                                int numRanks,
                                size_t localSize,
                                size_t n) {
  size_t k2 = blockIdx.x * blockDim.x + threadIdx.x;
  if (k2 < n) {
    size_t k = rank + k2 * numRanks;
    int dest = static_cast<int>(k / localSize);

    // Compute startK2: first k2 value that maps to this dest
    size_t startK2;
    if (dest == 0) {
      startK2 = 0;
    } else {
      size_t threshold = static_cast<size_t>(dest) * localSize;
      if (threshold <= static_cast<size_t>(rank)) {
        startK2 = 0;
      } else {
        startK2 = (threshold - rank + numRanks - 1) / numRanks;
      }
    }

    size_t packIdx = k2 - startK2;
    size_t outIdx = dest * chunk + packIdx;
    out[outIdx] = in[k2];
  }
}

/// Unpack after redistribution
template <typename ComplexT>
__global__ void fft1dUnpackKernel(const ComplexT* __restrict__ in,
                                  ComplexT* __restrict__ out,
                                  size_t chunk,
                                  int rank,
                                  int numRanks,
                                  size_t localSize,
                                  size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int src = static_cast<int>(idx / chunk);
    size_t pos = idx % chunk;

    // Reverse pack: find startK2 for data from src destined for us (rank)
    size_t startK2;
    size_t threshold = static_cast<size_t>(rank) * localSize;
    if (threshold <= static_cast<size_t>(src)) {
      startK2 = 0;
    } else {
      startK2 = (threshold - src + numRanks - 1) / numRanks;
    }

    size_t k2 = startK2 + pos;
    size_t k = src + k2 * numRanks;
    size_t outIdx = k % localSize;

    out[outIdx] = in[idx];
  }
}

//==============================================================================
// Cooley-Tukey Executor
//==============================================================================

/// Execute the complete 10-step Cooley-Tukey distributed FFT.
/// This encapsulates the entire PxP decomposition algorithm.
///
/// @tparam ComplexT   hipFloatComplex or hipDoubleComplex
/// @tparam RealT      float or double
/// @param dIn         Device input buffer
/// @param dOut        Device output buffer
/// @param bufs        Working buffers (dTemp, dBuf1, dBuf2, hSend, hRecv)
/// @param planP       hipFFT plan for P-point FFTs (batched)
/// @param planL       hipFFT plan for L-point FFT
/// @param localSize   L = fftLength / P
/// @param chunk       C = L / P
/// @param numRanks    P (number of MPI ranks)
/// @param rank        This rank's ID
/// @param fftLength   N' or M (the distributed FFT length)
/// @param direction   HIPFFT_FORWARD or HIPFFT_BACKWARD
/// @param mpiType     MPI_FLOAT or MPI_DOUBLE
/// @param stream      HIP stream for kernel execution
/// @param comm        MPI communicator
/// @return            hipfftResult (HIPFFT_SUCCESS on success)
template <typename ComplexT, typename RealT>
hipfftResult executeCooleyTukey(ComplexT* dIn,
                                ComplexT* dOut,
                                const FFT1DBuffers<ComplexT>& bufs,
                                hipfftHandle planP,
                                hipfftHandle planL,
                                size_t localSize,
                                size_t chunk,
                                size_t numRanks,
                                int rank,
                                size_t fftLength,
                                int direction,
                                MPI_Datatype mpiType,
                                hipStream_t stream,
                                MPI_Comm comm) {
  int mpiCount = static_cast<int>(chunk * 2);
  RealT twiddleSign =
      (direction == HIPFFT_FORWARD) ? static_cast<RealT>(-1.0) : static_cast<RealT>(1.0);

  // Step 1: First Alltoall
  alltoallWithStaging(dIn, bufs.dBuf1, bufs.hSend, bufs.hRecv, mpiCount, mpiType, localSize, comm);

  // Step 2: Scatter (transpose Pxchunk -> chunkxP)
  hipLaunchKernelGGL(fft1dScatterStep1Kernel<ComplexT>,
                     fft1dGridSize(localSize),
                     fft1dBlockSize(),
                     0,
                     stream,
                     bufs.dBuf1,
                     bufs.dBuf2,
                     chunk,
                     static_cast<int>(numRanks),
                     localSize);

  // Step 3: P-point FFTs
  hipfftResult fftRes;
  if constexpr (std::is_same_v<ComplexT, hipFloatComplex>) {
    fftRes = hipfftExecC2C(planP, bufs.dBuf2, bufs.dTemp, direction);
  } else {
    fftRes = hipfftExecZ2Z(planP, bufs.dBuf2, bufs.dTemp, direction);
  }
  if (fftRes != HIPFFT_SUCCESS)
    return fftRes;

  // Step 4: Twiddle factors
  hipLaunchKernelGGL((fft1dTwiddleKernel<ComplexT, RealT>),
                     fft1dGridSize(localSize),
                     fft1dBlockSize(),
                     0,
                     stream,
                     bufs.dTemp,
                     chunk,
                     static_cast<int>(numRanks),
                     rank,
                     fftLength,
                     twiddleSign,
                     localSize);

  // Step 5: Scatter (transpose chunkxP -> Pxchunk)
  hipLaunchKernelGGL(fft1dScatterStep4Kernel<ComplexT>,
                     fft1dGridSize(localSize),
                     fft1dBlockSize(),
                     0,
                     stream,
                     bufs.dTemp,
                     bufs.dBuf1,
                     chunk,
                     static_cast<int>(numRanks),
                     localSize);

  // Step 6: Second Alltoall
  alltoallWithStaging(
      bufs.dBuf1, bufs.dBuf2, bufs.hSend, bufs.hRecv, mpiCount, mpiType, localSize, comm);

  // Step 7: L-point FFT
  if constexpr (std::is_same_v<ComplexT, hipFloatComplex>) {
    fftRes = hipfftExecC2C(planL, bufs.dBuf2, bufs.dTemp, direction);
  } else {
    fftRes = hipfftExecZ2Z(planL, bufs.dBuf2, bufs.dTemp, direction);
  }
  if (fftRes != HIPFFT_SUCCESS)
    return fftRes;

  // Step 8: Pack for redistribution
  hipLaunchKernelGGL(fft1dPackKernel<ComplexT>,
                     fft1dGridSize(localSize),
                     fft1dBlockSize(),
                     0,
                     stream,
                     bufs.dTemp,
                     bufs.dBuf1,
                     chunk,
                     rank,
                     static_cast<int>(numRanks),
                     localSize,
                     localSize);

  // Step 9: Third Alltoall (with swapped host buffers)
  alltoallWithStagingSwapped(
      bufs.dBuf1, bufs.dBuf2, bufs.hSend, bufs.hRecv, mpiCount, mpiType, localSize, comm);

  // Step 10: Unpack to output
  hipLaunchKernelGGL(fft1dUnpackKernel<ComplexT>,
                     fft1dGridSize(localSize),
                     fft1dBlockSize(),
                     0,
                     stream,
                     bufs.dBuf2,
                     dOut,
                     chunk,
                     rank,
                     static_cast<int>(numRanks),
                     localSize,
                     localSize);

  return HIPFFT_SUCCESS;
}

/// In-place variant: output overwrites input buffer location
template <typename ComplexT, typename RealT>
hipfftResult executeCooleyTukeyInPlace(ComplexT* dInOut,
                                       const FFT1DBuffers<ComplexT>& bufs,
                                       hipfftHandle planP,
                                       hipfftHandle planL,
                                       size_t localSize,
                                       size_t chunk,
                                       size_t numRanks,
                                       int rank,
                                       size_t fftLength,
                                       int direction,
                                       MPI_Datatype mpiType,
                                       hipStream_t stream,
                                       MPI_Comm comm) {
  return executeCooleyTukey<ComplexT, RealT>(dInOut,
                                             dInOut,
                                             bufs,
                                             planP,
                                             planL,
                                             localSize,
                                             chunk,
                                             numRanks,
                                             rank,
                                             fftLength,
                                             direction,
                                             mpiType,
                                             stream,
                                             comm);
}

} // namespace shafft

#endif // SHAFFT_FFT1D_COOLEY_TUKEY_EXECUTOR_HIP
