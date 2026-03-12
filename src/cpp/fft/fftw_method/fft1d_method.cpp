#include "fft1d_method.hpp"
#include "common/env.hpp"
#include "common/normalize.hpp"

#include "../common/diag.hpp"

#include <shafft/shafft_error.hpp>

#include <fftw3-mpi.h>
#include <mpi.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <new>
#include <string>

namespace shafft {

using fft::diagEnabled;
using fft::mpiRank;

// Module-specific debug header
static void dbgHdr(MPI_Comm comm, const char* where) {
  fft::dbgHdr("1D-FFTW", comm, where);
}

static void ensureFFTwMpiInit() {
  static bool initialized = false;
  if (!initialized) {
    // Threading must be initialized before fftw_mpi_init
    int nthreads = fft::fftw::getenvThreadsOrDefault("SHAFFT_FFTW_THREADS", 1);
    if (nthreads > 1) {
      fftw_init_threads();
      fftwf_init_threads();
      fftw_plan_with_nthreads(nthreads);
      fftwf_plan_with_nthreads(nthreads);
    }

    fftw_mpi_init();
    fftwf_mpi_init();
    initialized = true;

    if (diagEnabled()) {
      dbgHdr(MPI_COMM_WORLD, "init");
      std::cerr << "FFTW-MPI initialized, threads=" << nthreads << "\n";
    }
  }
}

int fft1dQueryLayout(
    size_t globalN, size_t& localN, size_t& localStart, size_t& localAllocSize,
    size_t& localNTrans, size_t& localStartTrans, MPI_Comm comm) {
  ensureFFTwMpiInit();

  ptrdiff_t localNi, localStarti;
  ptrdiff_t localNo, localStarto;

  // First query on the original communicator to find inactive ranks.
  ptrdiff_t alloc = fftw_mpi_local_size_1d(
      static_cast<ptrdiff_t>(globalN),
      comm,
      FFTW_FORWARD,
      0,
      &localNi,
      &localStarti,
      &localNo,
      &localStarto);

  // If this rank has no local elements it is inactive.
  if (localNi == 0) {
    localN = 0;
    localStart = 0;
    localAllocSize = 0;
    localNTrans = 0;
    localStartTrans = 0;

    // MPI_Comm_split is collective; inactive ranks must enter it.
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm subcomm = MPI_COMM_NULL;
    int splitRc = MPI_Comm_split(comm, MPI_UNDEFINED, rank, &subcomm);
    if (splitRc != MPI_SUCCESS) {
      return static_cast<int>(Status::ERR_MPI);
    }
    return 0;
  }

  // Re-query on an active-rank subcommunicator.
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm subcomm = MPI_COMM_NULL;
  int splitRc = MPI_Comm_split(comm, 0, rank, &subcomm);
  if (splitRc != MPI_SUCCESS) {
    return static_cast<int>(Status::ERR_MPI);
  }

  ptrdiff_t subNi, subStarti, subNo, subStarto;
  ptrdiff_t subAlloc = fftw_mpi_local_size_1d(
      static_cast<ptrdiff_t>(globalN),
      subcomm,
      FFTW_FORWARD,
      0,
      &subNi,
      &subStarti,
      &subNo,
      &subStarto);
  MPI_Comm_free(&subcomm);

  localN = static_cast<size_t>(subNi);
  localStart = static_cast<size_t>(subStarti);
  localAllocSize = static_cast<size_t>(subAlloc);
  localNTrans = static_cast<size_t>(subNo);
  localStartTrans = static_cast<size_t>(subStarto);

  if (diagEnabled()) {
    dbgHdr(comm, "fft1dQueryLayout");
    std::cerr << "globalN=" << globalN
              << " localN=" << localN
              << " localStart=" << localStart
              << " localAllocSize=" << localAllocSize
              << " localNTrans=" << localNTrans
              << " localStartTrans=" << localStartTrans
              << "\n";
  }

  return 0;
}

int fft1dPlan(
    FFT1DHandle& handle, size_t globalN, FFTType precision, MPI_Comm comm, void* in, void* out) {
  ensureFFTwMpiInit();

  auto* internal = new (std::nothrow) FFTW1DInternal();
  if (!internal)
    return static_cast<int>(Status::ERR_ALLOC);

  internal->plan_fwd = nullptr;
  internal->plan_bwd = nullptr;
  internal->plan_fwd_sp = nullptr;
  internal->plan_bwd_sp = nullptr;
  internal->nthreads = fft::fftw::getenvThreadsOrDefault("SHAFFT_FFTW_THREADS", 1);
  internal->flags = fft::fftw::getenvPlannerFlags();

  ptrdiff_t localNi, localStarti;
  ptrdiff_t localNo, localStarto;
  ptrdiff_t allocSz;

  if (precision == FFTType::Z2Z) {
    allocSz = fftw_mpi_local_size_1d(static_cast<ptrdiff_t>(globalN),
                                     comm,
                                     FFTW_FORWARD,
                                     0,
                                     &localNi,
                                     &localStarti,
                                     &localNo,
                                     &localStarto);

    fftw_complex* bufIn = nullptr;
    fftw_complex* bufOut = nullptr;
    bool ownsBuffers = false;

    if (in && out) {
      bufIn = reinterpret_cast<fftw_complex*>(in);
      bufOut = reinterpret_cast<fftw_complex*>(out);
    } else {
      bufIn = fftw_alloc_complex(static_cast<size_t>(allocSz));
      bufOut = fftw_alloc_complex(static_cast<size_t>(allocSz));
      if (!bufIn || !bufOut) {
        if (bufIn)
          fftw_free(bufIn);
        if (bufOut)
          fftw_free(bufOut);
        delete internal;
        return static_cast<int>(Status::ERR_ALLOC);
      }
      ownsBuffers = true;
    }

    internal->plan_fwd = fftw_mpi_plan_dft_1d(
        static_cast<ptrdiff_t>(globalN), bufIn, bufOut, comm, FFTW_FORWARD, internal->flags);
    internal->plan_bwd = fftw_mpi_plan_dft_1d(
        static_cast<ptrdiff_t>(globalN), bufIn, bufOut, comm, FFTW_BACKWARD, internal->flags);

    if (ownsBuffers) {
      fftw_free(bufIn);
      fftw_free(bufOut);
    }

    if (!internal->plan_fwd || !internal->plan_bwd) {
      if (internal->plan_fwd)
        fftw_destroy_plan(internal->plan_fwd);
      if (internal->plan_bwd)
        fftw_destroy_plan(internal->plan_bwd);
      delete internal;
      return static_cast<int>(Status::ERR_BACKEND);
    }
  } else { // C2C
    allocSz = fftwf_mpi_local_size_1d(static_cast<ptrdiff_t>(globalN),
                                      comm,
                                      FFTW_FORWARD,
                                      0,
                                      &localNi,
                                      &localStarti,
                                      &localNo,
                                      &localStarto);

    // Use user-provided buffers if available, otherwise allocate temporary buffers.
    fftwf_complex* bufIn = nullptr;
    fftwf_complex* bufOut = nullptr;
    bool ownsBuffers = false;

    if (in && out) {
      bufIn = reinterpret_cast<fftwf_complex*>(in);
      bufOut = reinterpret_cast<fftwf_complex*>(out);
    } else {
      bufIn = fftwf_alloc_complex(static_cast<size_t>(allocSz));
      bufOut = fftwf_alloc_complex(static_cast<size_t>(allocSz));
      if (!bufIn || !bufOut) {
        if (bufIn)
          fftwf_free(bufIn);
        if (bufOut)
          fftwf_free(bufOut);
        delete internal;
        return static_cast<int>(Status::ERR_ALLOC);
      }
      ownsBuffers = true;
    }

    internal->plan_fwd_sp = fftwf_mpi_plan_dft_1d(
        static_cast<ptrdiff_t>(globalN), bufIn, bufOut, comm, FFTW_FORWARD, internal->flags);
    internal->plan_bwd_sp = fftwf_mpi_plan_dft_1d(
        static_cast<ptrdiff_t>(globalN), bufIn, bufOut, comm, FFTW_BACKWARD, internal->flags);

    if (ownsBuffers) {
      fftwf_free(bufIn);
      fftwf_free(bufOut);
    }

    if (!internal->plan_fwd_sp || !internal->plan_bwd_sp) {
      if (internal->plan_fwd_sp)
        fftwf_destroy_plan(internal->plan_fwd_sp);
      if (internal->plan_bwd_sp)
        fftwf_destroy_plan(internal->plan_bwd_sp);
      delete internal;
      return static_cast<int>(Status::ERR_BACKEND);
    }
  }

  handle.internal = internal;
  handle.globalN = globalN;
  handle.localNInit = static_cast<size_t>(localNi);
  handle.localStartInit = static_cast<size_t>(localStarti);
  handle.localNTrans = static_cast<size_t>(localNo);
  handle.localStartTrans = static_cast<size_t>(localStarto);
  handle.localAllocSize = static_cast<size_t>(allocSz);
  handle.precision = precision;
  handle.comm = comm;

  if (diagEnabled()) {
    dbgHdr(comm, "fft1dPlan");
    std::cerr << "globalN=" << globalN
              << " precision=" << (precision == FFTType::Z2Z ? "Z2Z" : "C2C")
              << " localNInit=" << handle.localNInit << " localNTrans=" << handle.localNTrans
              << " alloc=" << handle.localAllocSize << " threads=" << internal->nthreads
              << " flags="
              << (internal->flags == FFTW_ESTIMATE  ? "ESTIMATE"
                  : internal->flags == FFTW_MEASURE ? "MEASURE"
                  : internal->flags == FFTW_PATIENT ? "PATIENT"
                                                    : "EXHAUSTIVE")
              << "\n";
  }

  return 0;
}

int fft1dExecute(FFT1DHandle& handle, void* in, void* out, FFTDirection direction) {
  auto* internal = static_cast<FFTW1DInternal*>(handle.internal);
  if (!internal) {
    return static_cast<int>(Status::ERR_PLAN_NOT_INIT);
  }

  if (diagEnabled()) {
    dbgHdr(handle.comm, "fft1dExecute");
    std::cerr << (direction == FFTDirection::FORWARD ? "FORWARD" : "BACKWARD") << " in=" << in
              << " out=" << out << "\n";
  }

  if (handle.precision == FFTType::Z2Z) {
    fftw_plan p = (direction == FFTDirection::FORWARD) ? internal->plan_fwd : internal->plan_bwd;
    fftw_mpi_execute_dft(p, static_cast<fftw_complex*>(in), static_cast<fftw_complex*>(out));
  } else { // C2C
    fftwf_plan p =
        (direction == FFTDirection::FORWARD) ? internal->plan_fwd_sp : internal->plan_bwd_sp;
    fftwf_mpi_execute_dft(p, static_cast<fftwf_complex*>(in), static_cast<fftwf_complex*>(out));
  }

  return 0;
}

int fft1dNormalize(FFT1DHandle& handle, void* data, int normExponent, size_t localCount) {
  if (!data) {
    return static_cast<int>(Status::ERR_PLAN_NOT_INIT);
  }

  if (diagEnabled()) {
    dbgHdr(handle.comm, "fft1dNormalize");
    std::cerr << "globalN=" << handle.globalN << " localCount=" << localCount
              << " normExponent=" << normExponent << "\n";
  }

  if (handle.precision == FFTType::Z2Z) {
    double normFactor =
        1.0 / std::pow(std::sqrt(static_cast<double>(handle.globalN)), normExponent);
    return normalizeComplexDouble(normFactor, localCount, data);
  } else {
    float normFactor = 1.0f / std::pow(std::sqrt(static_cast<float>(handle.globalN)), normExponent);
    return normalizeComplexFloat(normFactor, localCount, data);
  }
}

int fft1dDestroy(FFT1DHandle& handle) {
  auto* internal = static_cast<FFTW1DInternal*>(handle.internal);
  if (!internal) {
    return 0; // Already destroyed
  }

  if (diagEnabled()) {
    dbgHdr(handle.comm, "fft1dDestroy");
    std::cerr << "globalN=" << handle.globalN << "\n";
  }

  if (internal->plan_fwd)
    fftw_destroy_plan(internal->plan_fwd);
  if (internal->plan_bwd)
    fftw_destroy_plan(internal->plan_bwd);
  if (internal->plan_fwd_sp)
    fftwf_destroy_plan(internal->plan_fwd_sp);
  if (internal->plan_bwd_sp)
    fftwf_destroy_plan(internal->plan_bwd_sp);

  // Note: We do NOT call fftw_mpi_cleanup() or fftw_cleanup_threads() here
  // because other plans may still be active. Call fft1dFinalize() at program
  // exit instead.

  delete internal;
  handle.internal = nullptr;

  return 0;
}

int fft1dFinalize() noexcept {
  fftw_mpi_cleanup();
  return 0;
}

} // namespace shafft
