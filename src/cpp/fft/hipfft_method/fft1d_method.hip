// fft1d_method.hip
// Distributed 1D FFT implementation using hipFFT backend.
//
// Supports two execution paths:
// - Path A (Cooley-Tukey): PxP decomposition, requires N' % P^2 == 0
// - Path B (Bluestein): Chirp-Z algorithm for arbitrary sizes
//
// Key features:
// - Cooley-Tukey 10-step sequence in reusable template function
// - C2C/Z2Z precision handled via templates
// - Buffer management abstracted into context structs

#include "1d/fft1d_bluestein_executor.hip"
#include "1d/fft1d_cooley_tukey_executor.hip"
#include "1d/fft1d_path_selector.hpp"
#include "common/normalize.hpp"
#include "fft1d_method.hpp"

#include "../common/diag.hpp"

#include "detail/error_macros.hpp"

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>
#include <mpi.h>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

// SHAFFT_GPU_AWARE_MPI provided by shafft_options compile definition

namespace shafft {

using fft::diagEnabled;
using fft::mpiRank;

// Module-specific debug header
static void dbgHdr(MPI_Comm comm, const char* where) {
  fft::dbgHdr("1D-hipfft", comm, where);
}

//==============================================================================
// Internal data structure
//==============================================================================

/// Unified internal state for both Cooley-Tukey and Bluestein paths.
/// Uses void* for buffers to support both C2C and Z2Z without templates at struct level.
struct FFT1dInternal {
  // Path selection
  FFT1DPath path;
  size_t globalN;   // True problem size (user's N)
  size_t fftLength; // N' (Path A) or M (Path B)

  // Common fields
  size_t localSize; // L = fftLength / P
  size_t numRanks;
  size_t chunk; // C = L / P
  int rank;

  // hipFFT plans
  hipfftHandle planP; // P-point FFTs (batched)
  hipfftHandle planL; // L-point FFT

  // Working buffers (allocated to localSize elements)
  void* dTemp;
  void* dBuf1;
  void* dBuf2;

  // Host staging buffers (non-GPU-aware MPI only)
  void* hSend;
  void* hRecv;

  // Bluestein-specific data (nullptr for Cooley-Tukey)
  void* dChirpB;    // Modulation chirp
  void* dChirpV;    // Convolution chirp (FFT domain)
  void* dModulated; // Working buffer

  // MPI
  hipStream_t stream;
};

//==============================================================================
// Helper functions
//==============================================================================

static size_t elementSize(FFTType precision) {
  return (precision == FFTType::C2C) ? sizeof(hipFloatComplex) : sizeof(hipDoubleComplex);
}

/// Compute block distribution: how many user elements this rank holds.
static void computeBlockDistributionWithL(
    size_t globalN, size_t L, int rank, size_t& localN, size_t& localStart) {
  size_t rankOffset = static_cast<size_t>(rank) * L;
  localStart = (rankOffset < globalN) ? rankOffset : globalN;

  if (rankOffset >= globalN) {
    localN = 0;
  } else {
    size_t remaining = globalN - rankOffset;
    localN = (remaining < L) ? remaining : L;
  }
}

//==============================================================================
// Template dispatch helpers
//==============================================================================

/// Create FFT1D buffer context from internal state
template <typename ComplexT>
FFT1DBuffers<ComplexT> makeCTBuffers(FFT1dInternal* internal) {
  FFT1DBuffers<ComplexT> bufs;
  bufs.dTemp = static_cast<ComplexT*>(internal->dTemp);
  bufs.dBuf1 = static_cast<ComplexT*>(internal->dBuf1);
  bufs.dBuf2 = static_cast<ComplexT*>(internal->dBuf2);
  bufs.hSend = static_cast<ComplexT*>(internal->hSend);
  bufs.hRecv = static_cast<ComplexT*>(internal->hRecv);
  return bufs;
}

/// Create Bluestein context from internal state and handle
template <typename ComplexT>
BluesteinContext<ComplexT>
makeBluesteinContext(FFT1dInternal* internal, size_t localN, size_t localStart) {
  BluesteinContext<ComplexT> ctx;
  ctx.dChirpB = static_cast<ComplexT*>(internal->dChirpB);
  ctx.dChirpV = static_cast<ComplexT*>(internal->dChirpV);
  ctx.dModulated = static_cast<ComplexT*>(internal->dModulated);
  ctx.ctBufs = makeCTBuffers<ComplexT>(internal);
  ctx.N = internal->globalN;
  ctx.M = internal->fftLength;
  ctx.localSize = internal->localSize;
  ctx.localN = localN;
  ctx.localStart = localStart;
  ctx.chunk = internal->chunk;
  return ctx;
}

//==============================================================================
// Execution dispatch
//==============================================================================

/// Execute FFT using the appropriate path with type dispatch.
template <typename ComplexT, typename RealT>
int executeTyped(FFT1DHandle& handle, ComplexT* dIn, ComplexT* dOut, FFTDirection direction) {
  FFT1dInternal* internal = static_cast<FFT1dInternal*>(handle.internal);

  const size_t localSize = internal->localSize;
  const size_t numRanks = internal->numRanks;
  const size_t chunk = internal->chunk;
  const int rank = internal->rank;
  const size_t fftLength = internal->fftLength;
  const hipStream_t stream = internal->stream;

  MPI_Datatype mpiType = std::is_same_v<ComplexT, hipFloatComplex> ? MPI_FLOAT : MPI_DOUBLE;
  int hipfftDir = (direction == FFTDirection::FORWARD) ? HIPFFT_FORWARD : HIPFFT_BACKWARD;
  int dirSign = (direction == FFTDirection::FORWARD) ? 1 : -1;

  auto ctBufs = makeCTBuffers<ComplexT>(internal);

  if (internal->path == FFT1DPath::BLUESTEIN) {
    // Bluestein path
    auto ctx = makeBluesteinContext<ComplexT>(internal, handle.localNInit, handle.localStartInit);

    return executeBluestein<ComplexT, RealT>(dIn,
                                             dOut,
                                             ctx,
                                             internal->planP,
                                             internal->planL,
                                             numRanks,
                                             rank,
                                             dirSign,
                                             mpiType,
                                             stream,
                                             handle.comm);
  } else {
    // Cooley-Tukey path

    // Zero-pad input beyond user data region
    // Needed for forward (padding N->N'), and also for backward if caller
    // supplies external frequency data that doesn't fill localSize.
    {
      const size_t userLocalN = handle.localNInit;
      if (userLocalN < localSize) {
        size_t padOffset = userLocalN * sizeof(ComplexT);
        size_t padBytes = (localSize - userLocalN) * sizeof(ComplexT);
        hipCheck(hipMemsetAsync(reinterpret_cast<char*>(dIn) + padOffset, 0, padBytes, stream));
        hipCheck(hipStreamSynchronize(stream));
      }
    }

    hipfftResult fftRes = executeCooleyTukey<ComplexT, RealT>(dIn,
                                                              dOut,
                                                              ctBufs,
                                                              internal->planP,
                                                              internal->planL,
                                                              localSize,
                                                              chunk,
                                                              numRanks,
                                                              rank,
                                                              fftLength,
                                                              hipfftDir,
                                                              mpiType,
                                                              stream,
                                                              handle.comm);

    return (fftRes == HIPFFT_SUCCESS) ? 0 : static_cast<int>(fftRes);
  }
}

//==============================================================================
// Resource allocation helpers
//==============================================================================

/// Allocate all device buffers for internal state
[[nodiscard]] static int
allocateDeviceBuffers(FFT1dInternal* internal, size_t localSize, size_t elemSz, bool isBluestein) {
  hipError_t err;

  err = hipMalloc(&internal->dTemp, localSize * elemSz);
  if (err != hipSuccess)
    return SHAFFT_STATUS(SHAFFT_ERR_ALLOC);

  err = hipMalloc(&internal->dBuf1, localSize * elemSz);
  if (err != hipSuccess) {
    hipCheck(hipFree(internal->dTemp));
    return SHAFFT_STATUS(SHAFFT_ERR_ALLOC);
  }

  err = hipMalloc(&internal->dBuf2, localSize * elemSz);
  if (err != hipSuccess) {
    hipCheck(hipFree(internal->dTemp));
    hipCheck(hipFree(internal->dBuf1));
    return SHAFFT_STATUS(SHAFFT_ERR_ALLOC);
  }

  if (isBluestein) {
    err = hipMalloc(&internal->dChirpB, localSize * elemSz);
    if (err != hipSuccess) {
      hipCheck(hipFree(internal->dTemp));
      hipCheck(hipFree(internal->dBuf1));
      hipCheck(hipFree(internal->dBuf2));
      return SHAFFT_STATUS(SHAFFT_ERR_ALLOC);
    }

    err = hipMalloc(&internal->dChirpV, localSize * elemSz);
    if (err != hipSuccess) {
      hipCheck(hipFree(internal->dTemp));
      hipCheck(hipFree(internal->dBuf1));
      hipCheck(hipFree(internal->dBuf2));
      hipCheck(hipFree(internal->dChirpB));
      return SHAFFT_STATUS(SHAFFT_ERR_ALLOC);
    }

    err = hipMalloc(&internal->dModulated, localSize * elemSz);
    if (err != hipSuccess) {
      hipCheck(hipFree(internal->dTemp));
      hipCheck(hipFree(internal->dBuf1));
      hipCheck(hipFree(internal->dBuf2));
      hipCheck(hipFree(internal->dChirpB));
      hipCheck(hipFree(internal->dChirpV));
      return SHAFFT_STATUS(SHAFFT_ERR_ALLOC);
    }
  } else {
    internal->dChirpB = nullptr;
    internal->dChirpV = nullptr;
    internal->dModulated = nullptr;
  }

  return SHAFFT_STATUS(SHAFFT_SUCCESS);
}

/// Free all device buffers
static void freeDeviceBuffers(FFT1dInternal* internal) {
  if (internal->dTemp)
    hipCheck(hipFree(internal->dTemp));
  if (internal->dBuf1)
    hipCheck(hipFree(internal->dBuf1));
  if (internal->dBuf2)
    hipCheck(hipFree(internal->dBuf2));
  if (internal->dChirpB)
    hipCheck(hipFree(internal->dChirpB));
  if (internal->dChirpV)
    hipCheck(hipFree(internal->dChirpV));
  if (internal->dModulated)
    hipCheck(hipFree(internal->dModulated));
}

/// Allocate host staging buffers
[[nodiscard]] static int
allocateHostBuffers(FFT1dInternal* internal, size_t localSize, size_t elemSz) {
#if SHAFFT_GPU_AWARE_MPI
  internal->hSend = nullptr;
  internal->hRecv = nullptr;
  return SHAFFT_STATUS(SHAFFT_SUCCESS);
#else
  hipError_t err;

  err = hipHostMalloc(&internal->hSend, localSize * elemSz, hipHostMallocDefault);
  if (err != hipSuccess)
    return SHAFFT_STATUS(SHAFFT_ERR_ALLOC);

  err = hipHostMalloc(&internal->hRecv, localSize * elemSz, hipHostMallocDefault);
  if (err != hipSuccess) {
    hipCheck(hipHostFree(internal->hSend));
    return SHAFFT_STATUS(SHAFFT_ERR_ALLOC);
  }

  return SHAFFT_STATUS(SHAFFT_SUCCESS);
#endif
}

/// Free host staging buffers
static void freeHostBuffers(FFT1dInternal* internal) {
#if !SHAFFT_GPU_AWARE_MPI
  if (internal->hSend)
    hipCheck(hipHostFree(internal->hSend));
  if (internal->hRecv)
    hipCheck(hipHostFree(internal->hRecv));
#endif
}

//==============================================================================
// Backend interface implementation
//==============================================================================

int fft1dQueryLayout(size_t globalN,
                     size_t& localN,
                     size_t& localStart,
                     size_t& localAllocSize,
                     size_t& localNTrans,
                     size_t& localStartTrans,
                     MPI_Comm comm,
                     const FFT1DConfig& config) {
  int rank, numRanks;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &numRanks);

  FFT1DPathConfig pathConfig =
      selectPath(globalN, static_cast<size_t>(numRanks), config.tau, config.allowPadding);
  size_t L = pathConfig.localSize;

  computeBlockDistributionWithL(globalN, L, rank, localN, localStart);
  // All ranks must allocate L elements: the distributed FFT algorithm
  // (both Cooley-Tukey and Bluestein) requires every rank to participate
  // in MPI_Alltoall with L-sized buffers, even if localN == 0.
  localAllocSize = L;

  // hipFFT uses symmetric layout: output distribution matches input.
  localNTrans = localN;
  localStartTrans = localStart;

  if (diagEnabled()) {
    dbgHdr(comm, "fft1dQueryLayout");
    std::cerr << "globalN=" << globalN << " path=" << pathToString(pathConfig.path)
              << " localN=" << localN << " localStart=" << localStart
              << " localAllocSize=" << localAllocSize
              << " localNTrans=" << localNTrans << " localStartTrans=" << localStartTrans << "\n";
  }

  return SHAFFT_STATUS(SHAFFT_SUCCESS);
}

int fft1dPlan(FFT1DHandle& handle,
              size_t globalN,
              FFTType precision,
              MPI_Comm comm,
              void* in,
              void* out,
              const FFT1DConfig& config) {
  (void)in;
  (void)out; // hipFFT doesn't need buffers at plan time

  int rank, rankCount;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &rankCount);
  size_t numRanks = static_cast<size_t>(rankCount);

  FFT1DPathConfig pathConfig = selectPath(globalN, numRanks, config.tau, config.allowPadding);
  bool isBluestein = (pathConfig.path == FFT1DPath::BLUESTEIN);

  if (diagEnabled()) {
    dbgHdr(comm, "fft1dPlan");
    std::cerr << "globalN=" << globalN << " path=" << pathToString(pathConfig.path)
              << " fftLength=" << pathConfig.fftLength << " localSize=" << pathConfig.localSize
              << " precision=" << (precision == FFTType::Z2Z ? "Z2Z" : "C2C") << "\n";
  }

  if (pathConfig.fftLength % numRanks != 0 || pathConfig.localSize % numRanks != 0) {
    return SHAFFT_STATUS(SHAFFT_ERR_INVALID_DIM);
  }

  // Allocate internal state
  FFT1dInternal* internal = new (std::nothrow) FFT1dInternal();
  if (!internal)
    return SHAFFT_STATUS(SHAFFT_ERR_ALLOC);

  internal->path = pathConfig.path;
  internal->globalN = globalN;
  internal->fftLength = pathConfig.fftLength;
  internal->localSize = pathConfig.localSize;
  internal->numRanks = numRanks;
  internal->chunk = isBluestein ? (pathConfig.localSize / numRanks) : pathConfig.chunk;
  internal->rank = rank;
  internal->stream = nullptr;

  size_t elemSz = elementSize(precision);
  size_t localSize = pathConfig.localSize;

  // Allocate buffers
  if (allocateDeviceBuffers(internal, localSize, elemSz, isBluestein) !=
      SHAFFT_STATUS(SHAFFT_SUCCESS)) {
    delete internal;
    return SHAFFT_STATUS(SHAFFT_ERR_ALLOC);
  }

  if (allocateHostBuffers(internal, localSize, elemSz) != SHAFFT_STATUS(SHAFFT_SUCCESS)) {
    freeDeviceBuffers(internal);
    delete internal;
    return SHAFFT_STATUS(SHAFFT_ERR_ALLOC);
  }

  // Create hipFFT plans
  hipfftType fftType = (precision == FFTType::C2C) ? HIPFFT_C2C : HIPFFT_Z2Z;
  size_t workSize = 0;

  // Guard against int overflow: hipfftMakePlan1d takes int for size/batch
  if (localSize > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      internal->chunk > static_cast<size_t>(std::numeric_limits<int>::max())) {
    freeDeviceBuffers(internal);
    freeHostBuffers(internal);
    delete internal;
    return SHAFFT_STATUS(SHAFFT_ERR_INVALID_DIM);
  }

  if (hipfftCreate(&internal->planP) != HIPFFT_SUCCESS ||
      hipfftMakePlan1d(internal->planP,
                       static_cast<int>(numRanks),
                       fftType,
                       static_cast<int>(internal->chunk),
                       &workSize) != HIPFFT_SUCCESS) {
    freeDeviceBuffers(internal);
    freeHostBuffers(internal);
    delete internal;
    return SHAFFT_STATUS(SHAFFT_ERR_BACKEND);
  }

  if (hipfftCreate(&internal->planL) != HIPFFT_SUCCESS ||
      hipfftMakePlan1d(internal->planL, static_cast<int>(localSize), fftType, 1, &workSize) !=
          HIPFFT_SUCCESS) {
    hipfftDestroy(internal->planP);
    freeDeviceBuffers(internal);
    freeHostBuffers(internal);
    delete internal;
    return SHAFFT_STATUS(SHAFFT_ERR_BACKEND);
  }

  // Precompute Bluestein chirps if needed
  if (isBluestein) {
    size_t localStart_M = static_cast<size_t>(rank) * localSize;
    size_t localN, localStart_N;
    computeBlockDistributionWithL(globalN, localSize, rank, localN, localStart_N);

    MPI_Datatype mpiType = (precision == FFTType::C2C) ? MPI_FLOAT : MPI_DOUBLE;
    int status = 0;

    if (precision == FFTType::C2C) {
      auto ctx = makeBluesteinContext<hipFloatComplex>(internal, localN, localStart_N);
      ctx.N = globalN;
      ctx.M = pathConfig.fftLength;
      // Copy back the context pointers we need
      internal->dChirpB = ctx.dChirpB;
      internal->dChirpV = ctx.dChirpV;

      status = precomputeBluesteinChirps<hipFloatComplex, float>(ctx,
                                                                 internal->planP,
                                                                 internal->planL,
                                                                 numRanks,
                                                                 rank,
                                                                 localStart_M,
                                                                 localStart_N,
                                                                 mpiType,
                                                                 nullptr,
                                                                 comm);
    } else {
      auto ctx = makeBluesteinContext<hipDoubleComplex>(internal, localN, localStart_N);
      ctx.N = globalN;
      ctx.M = pathConfig.fftLength;
      internal->dChirpB = ctx.dChirpB;
      internal->dChirpV = ctx.dChirpV;

      status = precomputeBluesteinChirps<hipDoubleComplex, double>(ctx,
                                                                   internal->planP,
                                                                   internal->planL,
                                                                   numRanks,
                                                                   rank,
                                                                   localStart_M,
                                                                   localStart_N,
                                                                   mpiType,
                                                                   nullptr,
                                                                   comm);
    }

    if (status != 0) {
      hipfftDestroy(internal->planP);
      hipfftDestroy(internal->planL);
      freeDeviceBuffers(internal);
      freeHostBuffers(internal);
      delete internal;
      return SHAFFT_STATUS(SHAFFT_ERR_BACKEND);
    }
  }

  // Setup handle
  handle.internal = internal;
  handle.globalN = globalN;

  size_t userLocalN, userLocalStart;
  computeBlockDistributionWithL(globalN, localSize, rank, userLocalN, userLocalStart);
  handle.localNInit = userLocalN;
  handle.localStartInit = userLocalStart;
  handle.localNTrans = userLocalN;
  handle.localStartTrans = userLocalStart;
  handle.localAllocSize = localSize;
  handle.precision = precision;
  handle.comm = comm;

  return SHAFFT_STATUS(SHAFFT_SUCCESS);
}

int fft1dDestroy(FFT1DHandle& handle) {
  if (!handle.internal)
    return SHAFFT_STATUS(SHAFFT_SUCCESS);

  FFT1dInternal* internal = static_cast<FFT1dInternal*>(handle.internal);

  hipfftDestroy(internal->planP);
  hipfftDestroy(internal->planL);
  freeDeviceBuffers(internal);
  freeHostBuffers(internal);

  delete internal;
  handle.internal = nullptr;

  return SHAFFT_STATUS(SHAFFT_SUCCESS);
}

int fft1dSetStream(FFT1DHandle& handle, hipStream_t stream) {
  if (!handle.internal)
    return SHAFFT_STATUS(SHAFFT_ERR_PLAN_NOT_INIT);

  FFT1dInternal* internal = static_cast<FFT1dInternal*>(handle.internal);
  internal->stream = stream;
  handle.stream = stream;

  if (hipfftCheck(hipfftSetStream(internal->planP, stream))) {
    return SHAFFT_STATUS(SHAFFT_ERR_BACKEND);
  }
  if (hipfftCheck(hipfftSetStream(internal->planL, stream))) {
    return SHAFFT_STATUS(SHAFFT_ERR_BACKEND);
  }

  return SHAFFT_STATUS(SHAFFT_SUCCESS);
}

int fft1dSynchronize(FFT1DHandle& handle) {
  if (!handle.internal)
    return SHAFFT_STATUS(SHAFFT_ERR_PLAN_NOT_INIT);

  FFT1dInternal* internal = static_cast<FFT1dInternal*>(handle.internal);
  hipError_t err = hipStreamSynchronize(internal->stream);

  return (err == hipSuccess) ? SHAFFT_STATUS(SHAFFT_SUCCESS) : SHAFFT_STATUS(SHAFFT_ERR_BACKEND);
}

int fft1dExecute(FFT1DHandle& handle, void* in, void* out, FFTDirection direction) {
  if (!handle.internal)
    return SHAFFT_STATUS(SHAFFT_ERR_PLAN_NOT_INIT);

  FFT1dInternal* internal = static_cast<FFT1dInternal*>(handle.internal);

  if (diagEnabled()) {
    dbgHdr(handle.comm, "fft1dExecute");
    std::cerr << (direction == FFTDirection::FORWARD ? "FORWARD" : "BACKWARD")
              << " N=" << handle.globalN << " path=" << pathToString(internal->path) << "\n";
  }

  int rc;
  if (handle.precision == FFTType::C2C) {
    rc = executeTyped<hipFloatComplex, float>(
        handle, static_cast<hipFloatComplex*>(in), static_cast<hipFloatComplex*>(out), direction);
  } else {
    rc = executeTyped<hipDoubleComplex, double>(
        handle, static_cast<hipDoubleComplex*>(in), static_cast<hipDoubleComplex*>(out), direction);
  }

  return (rc == 0) ? SHAFFT_STATUS(SHAFFT_SUCCESS) : SHAFFT_STATUS(SHAFFT_ERR_BACKEND);
}

int fft1dNormalize(FFT1DHandle& handle, void* data, int normExponent, size_t localCount) {
  if (!handle.internal)
    return SHAFFT_STATUS(SHAFFT_ERR_PLAN_NOT_INIT);

  FFT1dInternal* internal = static_cast<FFT1dInternal*>(handle.internal);
  const size_t nTrue = handle.globalN;

  if (diagEnabled()) {
    dbgHdr(handle.comm, "fft1dNormalize");
    std::cerr << "globalN=" << nTrue << " localCount=" << localCount
              << " normExponent=" << normExponent << "\n";
  }

  int rc = 0;
  if (handle.precision == FFTType::C2C) {
    float norm = 1.0f / std::pow(std::sqrt(static_cast<float>(nTrue)), normExponent);
    rc = normalizeComplexFloat(norm, localCount, data, internal->stream);
  } else {
    double norm = 1.0 / std::pow(std::sqrt(static_cast<double>(nTrue)), normExponent);
    rc = normalizeComplexDouble(norm, localCount, data, internal->stream);
  }

  if (rc != 0)
    return SHAFFT_STATUS(SHAFFT_ERR_BACKEND);

  hipCheck(hipStreamSynchronize(internal->stream));
  return SHAFFT_STATUS(SHAFFT_SUCCESS);
}

} // namespace shafft
