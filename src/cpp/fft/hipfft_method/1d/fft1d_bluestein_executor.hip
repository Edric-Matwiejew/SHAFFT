// fft1d_bluestein_executor.hip
// Path B: Bluestein (chirp-Z) distributed 1D FFT implementation.
// Contains GPU kernels, chirp helpers, and the executor function.
//
// Converts arbitrary N-point DFT to convolution via FFT of size M.
// Uses Cooley-Tukey internally for the M-point distributed FFTs.
//
// Algorithm:
//   1. Modulate: a[n] = x[n] * b[n]
//   2. Forward FFT of modulated signal (M-point distributed via Cooley-Tukey)
//   3. Pointwise multiply with precomputed V_hat (convolution in frequency domain)
//   4. Inverse FFT (M-point distributed via Cooley-Tukey)
//   5. Scale by 1/M (IFFT normalization)
//   6. Demodulate: X[k] = c[k] * b[k]

#ifndef SHAFFT_FFT1D_BLUESTEIN_EXECUTOR_HIP
#define SHAFFT_FFT1D_BLUESTEIN_EXECUTOR_HIP

#include "fft1d_common.hip"
#include "fft1d_cooley_tukey_executor.hip"

#include <cmath>
#include <cstdlib>

namespace shafft {

//==============================================================================
// GPU Kernels for Bluestein
//==============================================================================

/// Chirp modulation: a[n] = x[n] * b[n] for forward, a[n] = conj(x[n]) * b[n] for backward
/// chirp_B contains b[n] = exp(-i*pi*n^2/N)
///
/// For backward, we use IFFT(x) = conj(FFT(conj(x)))/N identity:
///   - Conjugate input BEFORE modulation
///   - Use the same forward chirp b (not b*)
///   - Output conjugation happens in demodulate kernel
template <typename ComplexT, typename RealT>
__global__ void
bluesteinModulateKernel(const ComplexT* __restrict__ x,
                        const ComplexT* __restrict__ chirpB,
                        ComplexT* __restrict__ a,
                        size_t N,           // True problem size
                        size_t localN,      // Elements this rank actually has
                        size_t localOffset, // Global index offset for this rank in N-space
                        size_t localSize,   // Local allocation size (M/P)
                        int direction)      // 1 for forward, -1 for backward
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < localSize) {
    size_t globalIdx = localOffset + idx; // Global index in N-space

    if (globalIdx < N && idx < localN) {
      // Within user data
      ComplexT xv = x[idx];
      ComplexT cv = chirpB[idx]; // Always use forward chirp b

      if (direction < 0) {
        // Backward: conjugate input first (part of IFFT = conj(FFT(conj(x)))/N)
        xv.y = -xv.y;
      }

      // Multiply: a = xv * cv
      a[idx].x = xv.x * cv.x - xv.y * cv.y;
      a[idx].y = xv.x * cv.y + xv.y * cv.x;
    } else {
      // Zero-pad
      a[idx].x = static_cast<RealT>(0);
      a[idx].y = static_cast<RealT>(0);
    }
  }
}

/// Chirp demodulation
/// Forward: X[k] = c[k] * b[k]
/// Backward: X[k] = conj(c[k] * b[k])  (final conjugation for IFFT identity)
///
/// For backward, we complete IFFT(x) = conj(FFT(conj(x)))/N:
///   - Use the same forward chirp b (not b*)
///   - Conjugate the OUTPUT after demodulation
template <typename ComplexT, typename RealT>
__global__ void bluesteinDemodulateKernel(const ComplexT* __restrict__ c,
                                          const ComplexT* __restrict__ chirpB,
                                          ComplexT* __restrict__ X,
                                          size_t N,           // True problem size
                                          size_t localN,      // Elements this rank actually outputs
                                          size_t localOffset, // Global index offset for this rank
                                          size_t localSize,   // Local allocation size
                                          int direction)      // 1 for forward, -1 for backward
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < localSize) {
    size_t globalIdx = localOffset + idx;

    if (globalIdx < N && idx < localN) {
      ComplexT cv = c[idx];
      ComplexT chirpv = chirpB[idx]; // Always use forward chirp b

      // Demodulate: result = cv * chirpv
      RealT rx = cv.x * chirpv.x - cv.y * chirpv.y;
      RealT ry = cv.x * chirpv.y + cv.y * chirpv.x;

      if (direction < 0) {
        // Backward: conjugate output (final step of IFFT = conj(FFT(conj(x)))/N)
        ry = -ry;
      }

      X[idx].x = rx;
      X[idx].y = ry;
    } else {
      // Zero beyond N
      X[idx].x = static_cast<RealT>(0);
      X[idx].y = static_cast<RealT>(0);
    }
  }
}

/// Pointwise multiply: C[k] = A[k] * B[k]
/// Used for convolution in frequency domain
template <typename ComplexT>
__global__ void bluesteinPointwiseMultiplyKernel(ComplexT* __restrict__ A,
                                                 const ComplexT* __restrict__ B,
                                                 size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    ComplexT av = A[idx];
    ComplexT bv = B[idx];
    A[idx].x = av.x * bv.x - av.y * bv.y;
    A[idx].y = av.x * bv.y + av.y * bv.x;
  }
}

/// Scale kernel for 1/M normalization after inverse FFT in convolution
template <typename ComplexT, typename RealT>
__global__ void bluesteinScaleKernel(ComplexT* __restrict__ data, RealT scale, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx].x *= scale;
    data[idx].y *= scale;
  }
}

//==============================================================================
// Host-side chirp precomputation
//==============================================================================

/// Precompute chirp sequence b[n] = exp(-i * pi * n^2 / N) for n in local range
/// Forward DFT modulates with b[n], demodulates with b[n] (same chirp)
template <typename ComplexT, typename RealT>
void computeLocalChirpB(ComplexT* chirpB, size_t N, size_t localStart, size_t localCount) {
  for (size_t i = 0; i < localCount; ++i) {
    size_t n = localStart + i;
    if (n < N) {
      // angle = -pi * n^2 / N (negative sign for Bluestein chirp)
      // Cast to double before squaring to avoid overflow for large n
      double nD = static_cast<double>(n);
      double ND = static_cast<double>(N);
      double angle = -M_PI * nD * nD / ND;
      chirpB[i].x = static_cast<RealT>(std::cos(angle));
      chirpB[i].y = static_cast<RealT>(std::sin(angle));
    } else {
      // Zero for indices beyond N
      chirpB[i].x = static_cast<RealT>(0);
      chirpB[i].y = static_cast<RealT>(0);
    }
  }
}

/// Precompute convolution chirp V for local portion
/// V is the FFT of the padded chirp sequence:
/// v[m] = conj(b[m]) for m = 0..N-1
/// v[m] = 0 for m = N..M-N
/// v[m] = conj(b[M-m]) for m = M-N+1..M-1
/// Since b[n] = exp(-ipin^2/N), conj(b[n]) = exp(+ipin^2/N)
template <typename ComplexT, typename RealT>
void computeLocalConvolutionChirp(
    ComplexT* v_local, size_t N, size_t M, size_t localStart, size_t localCount) {
  double ND = static_cast<double>(N);
  for (size_t i = 0; i < localCount; ++i) {
    size_t m = localStart + i;
    double angle = 0;
    bool hasValue = false;

    if (m < N) {
      // v[m] = conj(b[m]) = exp(+i*pi*m^2/N)
      // Cast to double before squaring to avoid overflow
      double mD = static_cast<double>(m);
      angle = M_PI * mD * mD / ND;
      hasValue = true;
    } else if (m > M - N && m < M) {
      // v[m] = conj(b[M-m]) for wrap-around
      double idxD = static_cast<double>(M - m);
      angle = M_PI * idxD * idxD / ND;
      hasValue = true;
    }

    if (hasValue) {
      v_local[i].x = static_cast<RealT>(std::cos(angle));
      v_local[i].y = static_cast<RealT>(std::sin(angle));
    } else {
      v_local[i].x = static_cast<RealT>(0);
      v_local[i].y = static_cast<RealT>(0);
    }
  }
}

//==============================================================================
// Bluestein Executor
//==============================================================================

/// Context for Bluestein execution
template <typename ComplexT>
struct BluesteinContext {
  // Chirp sequences (precomputed during plan creation)
  ComplexT* dChirpB;    // b[n] = exp(-i*pi*n^2/N) for modulation
  ComplexT* dChirpV;    // Precomputed FFT of convolution chirp
  ComplexT* dModulated; // Working buffer for modulated signal

  // Cooley-Tukey buffers for internal FFTs
  FFT1DBuffers<ComplexT> ctBufs;

  // Problem sizes
  size_t N;          // True problem size
  size_t M;          // Convolution length (M >= 2N-1, M % P^2 == 0)
  size_t localSize;  // M / P
  size_t localN;     // User elements this rank has
  size_t localStart; // User element offset in N-space
  size_t chunk;      // localSize / P
};

/// Execute the complete Bluestein algorithm for distributed FFT.
/// Uses Cooley-Tukey internally for the M-point FFTs.
///
/// @tparam ComplexT   hipFloatComplex or hipDoubleComplex
/// @tparam RealT      float or double
/// @param dIn         Device input buffer (user data, size localN)
/// @param dOut        Device output buffer
/// @param ctx         Bluestein context with chirps and buffers
/// @param planP       hipFFT plan for P-point FFTs
/// @param planL       hipFFT plan for L-point FFT
/// @param numRanks    P (number of MPI ranks)
/// @param rank        This rank's ID
/// @param direction   FFT direction (1 = forward, -1 = backward)
/// @param mpiType     MPI_FLOAT or MPI_DOUBLE
/// @param stream      HIP stream
/// @param comm        MPI communicator
/// @return            0 on success, non-zero on error
template <typename ComplexT, typename RealT>
int executeBluestein(ComplexT* dIn,
                     ComplexT* dOut,
                     BluesteinContext<ComplexT>& ctx,
                     hipfftHandle planP,
                     hipfftHandle planL,
                     size_t numRanks,
                     int rank,
                     int direction,
                     MPI_Datatype mpiType,
                     hipStream_t stream,
                     MPI_Comm comm) {
  const size_t localSize = ctx.localSize;
  const size_t chunk = ctx.chunk;
  const size_t N = ctx.N;
  const size_t M = ctx.M;
  const size_t localN = ctx.localN;
  const size_t localStart = ctx.localStart;

  // Step B1: Modulate input with chirp b
  hipLaunchKernelGGL((bluesteinModulateKernel<ComplexT, RealT>),
                     fft1dGridSize(localSize),
                     fft1dBlockSize(),
                     0,
                     stream,
                     dIn,
                     ctx.dChirpB,
                     ctx.dModulated,
                     N,
                     localN,
                     localStart,
                     localSize,
                     direction);

  // Step B2: Forward FFT of modulated signal (distributed M-point FFT)
  hipfftResult fftRes =
      executeCooleyTukey<ComplexT, RealT>(ctx.dModulated,
                                          ctx.dModulated, // In-place to dModulated
                                          ctx.ctBufs,
                                          planP,
                                          planL,
                                          localSize,
                                          chunk,
                                          numRanks,
                                          rank,
                                          M,
                                          HIPFFT_FORWARD,
                                          mpiType,
                                          stream,
                                          comm);
  if (fftRes != HIPFFT_SUCCESS)
    return static_cast<int>(fftRes);

  // Step B3: Pointwise multiply with precomputed V_hat
  hipLaunchKernelGGL(bluesteinPointwiseMultiplyKernel<ComplexT>,
                     fft1dGridSize(localSize),
                     fft1dBlockSize(),
                     0,
                     stream,
                     ctx.dModulated,
                     ctx.dChirpV,
                     localSize);

  // Step B4: Inverse FFT (distributed M-point IFFT)
  fftRes = executeCooleyTukey<ComplexT, RealT>(ctx.dModulated,
                                               ctx.dModulated,
                                               ctx.ctBufs,
                                               planP,
                                               planL,
                                               localSize,
                                               chunk,
                                               numRanks,
                                               rank,
                                               M,
                                               HIPFFT_BACKWARD,
                                               mpiType,
                                               stream,
                                               comm);
  if (fftRes != HIPFFT_SUCCESS)
    return static_cast<int>(fftRes);

  // Step B5: Scale by 1/M (IFFT normalization for convolution)
  RealT scale = static_cast<RealT>(1.0) / static_cast<RealT>(M);
  hipLaunchKernelGGL((bluesteinScaleKernel<ComplexT, RealT>),
                     fft1dGridSize(localSize),
                     fft1dBlockSize(),
                     0,
                     stream,
                     ctx.dModulated,
                     scale,
                     localSize);

  // Step B6: Demodulate with chirp to get final result
  hipLaunchKernelGGL((bluesteinDemodulateKernel<ComplexT, RealT>),
                     fft1dGridSize(localSize),
                     fft1dBlockSize(),
                     0,
                     stream,
                     ctx.dModulated,
                     ctx.dChirpB,
                     dOut,
                     N,
                     localN,
                     localStart,
                     localSize,
                     direction);

  return 0;
}

//==============================================================================
// Bluestein chirp precomputation
//==============================================================================

/// Precompute chirp sequences for Bluestein algorithm.
/// Computes chirpB (modulation chirp) and chirpV (FFT of convolution chirp).
///
/// @tparam ComplexT   hipFloatComplex or hipDoubleComplex
/// @tparam RealT      float or double
/// @param ctx         Bluestein context (dChirpB, dChirpV must be allocated)
/// @param planP       hipFFT plan for P-point FFTs
/// @param planL       hipFFT plan for L-point FFT
/// @param numRanks    P (number of MPI ranks)
/// @param rank        This rank's ID
/// @param localStart_M  M-space offset for this rank
/// @param localStart_N  N-space offset for user data
/// @param mpiType     MPI datatype (MPI_FLOAT or MPI_DOUBLE)
/// @param stream      HIP stream (can be nullptr for synchronous)
/// @param comm        MPI communicator
/// @return            0 on success, non-zero on error
template <typename ComplexT, typename RealT>
int precomputeBluesteinChirps(BluesteinContext<ComplexT>& ctx,
                              hipfftHandle planP,
                              hipfftHandle planL,
                              size_t numRanks,
                              int rank,
                              size_t localStart_M,
                              size_t localStart_N,
                              MPI_Datatype mpiType,
                              hipStream_t stream,
                              MPI_Comm comm) {
  const size_t N = ctx.N;
  const size_t M = ctx.M;
  const size_t localSize = ctx.localSize;
  const size_t chunk = ctx.chunk;

  // Allocate host buffers for chirp computation
  void* hChirpB = malloc(localSize * sizeof(ComplexT));
  void* hChirpV = malloc(localSize * sizeof(ComplexT));

  // RAII-style cleanup using a simple scope guard
  auto cleanup = [&]() {
    free(hChirpB);
    free(hChirpV);
  };

  if (!hChirpB || !hChirpV) {
    cleanup();
    return -1; // Allocation error
  }

  // Compute local portions of chirp sequences on host
  computeLocalChirpB<ComplexT, RealT>(static_cast<ComplexT*>(hChirpB), N, localStart_N, localSize);
  computeLocalConvolutionChirp<ComplexT, RealT>(
      static_cast<ComplexT*>(hChirpV), N, M, localStart_M, localSize);

  // Copy to device
  hipError_t err =
      hipMemcpy(ctx.dChirpB, hChirpB, localSize * sizeof(ComplexT), hipMemcpyHostToDevice);
  if (err != hipSuccess) {
    cleanup();
    return static_cast<int>(err);
  }

  err = hipMemcpy(ctx.dChirpV, hChirpV, localSize * sizeof(ComplexT), hipMemcpyHostToDevice);
  if (err != hipSuccess) {
    cleanup();
    return static_cast<int>(err);
  }

  // Done with host buffers - free them before the FFT call
  cleanup();

  // Now FFT the convolution chirp V to get V_hat in frequency domain
  // Uses the Cooley-Tukey executor for the distributed M-point FFT
  hipfftResult fftRes = executeCooleyTukeyInPlace<ComplexT, RealT>(ctx.dChirpV,
                                                                   ctx.ctBufs,
                                                                   planP,
                                                                   planL,
                                                                   localSize,
                                                                   chunk,
                                                                   numRanks,
                                                                   rank,
                                                                   M,
                                                                   HIPFFT_FORWARD,
                                                                   mpiType,
                                                                   stream,
                                                                   comm);

  return (fftRes == HIPFFT_SUCCESS) ? 0 : static_cast<int>(fftRes);
}

} // namespace shafft

#endif // SHAFFT_FFT1D_BLUESTEIN_EXECUTOR_HIP
