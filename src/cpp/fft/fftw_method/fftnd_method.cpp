// Environment variables:
//   SHAFFT_FFTW_THREADS  - Number of threads for FFTW (default: 1)
//   SHAFFT_FFTW_PLANNER  - Planner strategy: ESTIMATE (default), MEASURE, PATIENT, EXHAUSTIVE
//   SHAFFT_FFT_DIAG      - Set to "1" to enable diagnostic output

#include "fftnd_method.hpp"

#include <shafft/shafft_types.hpp>

#include "../../detail/array_utils.hpp"
#include "../common/diag.hpp"
#include "common/env.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <fftw3.h>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using shafft::fft::diagEnabled;
using shafft::fft::mpiRank;

// Module-specific debug header
static void dbgHdr(MPI_Comm comm, const char* where) {
  shafft::fft::dbgHdr("ND-FFTW", comm, where);
}

static void buildGuru64Dims(int ndim,
                            const int* dims,
                            int a,
                            int b,
                            std::vector<fftw_iodim64>& t,
                            std::vector<fftw_iodim64>& blist) {
  std::vector<long long> strides(ndim + 1, 1);
  for (int m = ndim - 1; m >= 0; --m)
    strides[m] = strides[m + 1] * static_cast<long long>(dims[m]);

  // Transform dims: contiguous block [a..b]
  const int r = b - a + 1;
  t.resize(r);
  for (int k = 0; k < r; ++k) {
    const int m = a + k;
    t[k].n = (ptrdiff_t)dims[m];
    t[k].is = (ptrdiff_t)strides[m + 1];
    t[k].os = (ptrdiff_t)strides[m + 1];
  }

  // Batch dims: all axes except [a..b]
  blist.clear();
  blist.reserve(ndim - r);
  for (int m = 0; m < ndim; ++m) {
    if (m >= a && m <= b)
      continue;
    fftw_iodim64 d;
    d.n = (ptrdiff_t)dims[m];
    d.is = (ptrdiff_t)strides[m + 1];
    d.os = (ptrdiff_t)strides[m + 1];
    blist.push_back(d);
  }
}

// ---------------- API ----------------

int fftndPlan(FFTNDHandle& plan,
              int nta,
              int* ta,
              int ndim,
              int* dimensions,
              shafft::FFTType precision,
              void* in,
              void* out) {
  plan.fft_type = precision;
  plan.ndim = ndim;
  plan.nta = nta;

  if (diagEnabled()) {
    dbgHdr(plan.comm, "fftndPlan");
    std::cerr << "ndim=" << ndim << " nta=" << nta
              << " precision=" << (precision == shafft::FFTType::Z2Z ? "Z2Z" : "C2C") << " dims=[";
    for (int i = 0; i < ndim; ++i)
      std::cerr << (i ? "," : "") << dimensions[i];
    std::cerr << "] ta=[";
    for (int i = 0; i < nta; ++i)
      std::cerr << (i ? "," : "") << ta[i];
    std::cerr << "]\n";
  }

  if (ndim <= 0)
    throw std::invalid_argument("FFTW: ndim must be > 0");
  for (int i = 0; i < ndim; ++i)
    if (dimensions[i] <= 0)
      throw std::invalid_argument("FFTW: dimensions must be > 0");

  // Handle degenerate stage (no local axes to transform)
  if (nta == 0) {
    plan.nsubplans = 0; // upstream will detect even parity -> no swap
    if (diagEnabled()) {
      dbgHdr(plan.comm, "fftndPlan");
      std::cerr << "nta=0, degenerate stage\n";
    }
    return 0;
  }

  // Find contiguous block [a..b] from ta[]
  int a = ta[0], b = ta[0];
  for (int i = 1; i < nta; ++i) {
    a = std::min(a, ta[i]);
    b = std::max(b, ta[i]);
  }
  if (b - a + 1 != nta)
    throw std::invalid_argument("FFTW: transform axes must form a contiguous block");
  plan.a = a;
  plan.b = b;

  std::vector<fftw_iodim64> t, blist;
  buildGuru64Dims(ndim, dimensions, a, b, t, blist);

  plan.nthreads = shafft::fft::fftw::getenvThreadsOrDefault("SHAFFT_FFTW_THREADS", 1);
  if (precision == shafft::FFTType::Z2Z) {
    fftw_init_threads();
    fftw_plan_with_nthreads(plan.nthreads);
  } else {
    fftwf_init_threads();
    fftwf_plan_with_nthreads(plan.nthreads);
  }

  plan.flags = shafft::fft::fftw::getenvPlannerFlags();

  // Use user-provided buffers if available, otherwise allocate temporary buffers for planning.
  const size_t totalElems = shafft::detail::prodN<int, size_t>(dimensions, ndim);
  fftw_complex* bufDp = nullptr;
  fftwf_complex* bufSp = nullptr;
  bool ownsBuffer = false;

  if (precision == shafft::FFTType::Z2Z) {
    if (in) {
      bufDp = reinterpret_cast<fftw_complex*>(in);
    } else {
      bufDp = (fftw_complex*)fftw_malloc(totalElems * sizeof(fftw_complex));
      if (!bufDp)
        throw std::runtime_error("FFTW: buffer allocation failed (double)");
      ownsBuffer = true;
    }
  } else {
    if (in) {
      bufSp = reinterpret_cast<fftwf_complex*>(in);
    } else {
      bufSp = (fftwf_complex*)fftwf_malloc(totalElems * sizeof(fftwf_complex));
      if (!bufSp)
        throw std::runtime_error("FFTW: buffer allocation failed (float)");
      ownsBuffer = true;
    }
  }

  (void)out;

  if (precision == shafft::FFTType::Z2Z) {
    plan.fwd_dp = fftw_plan_guru64_dft((int)t.size(),
                                       t.data(),
                                       (int)blist.size(),
                                       blist.data(),
                                       bufDp,
                                       bufDp,
                                       FFTW_FORWARD,
                                       plan.flags);
    plan.bwd_dp = fftw_plan_guru64_dft((int)t.size(),
                                       t.data(),
                                       (int)blist.size(),
                                       blist.data(),
                                       bufDp,
                                       bufDp,
                                       FFTW_BACKWARD,
                                       plan.flags);
    if (ownsBuffer)
      fftw_free(bufDp);
    if (!plan.fwd_dp || !plan.bwd_dp)
      throw std::runtime_error("FFTW: plan creation failed (double)");
  } else { // single precision
    std::vector<fftwf_iodim64> tf(t.size()), bf(blist.size());
    for (size_t i = 0; i < t.size(); ++i) {
      tf[i].n = t[i].n;
      tf[i].is = t[i].is;
      tf[i].os = t[i].os;
    }
    for (size_t i = 0; i < blist.size(); ++i) {
      bf[i].n = blist[i].n;
      bf[i].is = blist[i].is;
      bf[i].os = blist[i].os;
    }

    plan.fwd_sp = fftwf_plan_guru64_dft((int)tf.size(),
                                        tf.data(),
                                        (int)bf.size(),
                                        bf.data(),
                                        bufSp,
                                        bufSp,
                                        FFTW_FORWARD,
                                        plan.flags);
    plan.bwd_sp = fftwf_plan_guru64_dft((int)tf.size(),
                                        tf.data(),
                                        (int)bf.size(),
                                        bf.data(),
                                        bufSp,
                                        bufSp,
                                        FFTW_BACKWARD,
                                        plan.flags);
    if (ownsBuffer)
      fftwf_free(bufSp);
    if (!plan.fwd_sp || !plan.bwd_sp)
      throw std::runtime_error("FFTW: plan creation failed (single)");
  }

  plan.nsubplans = 1; // odd so upstream swaps buffers once

  if (diagEnabled()) {
    dbgHdr(plan.comm, "fftndPlan");
    std::cerr << "created guru64 plan: axes=[" << a << ".." << b << "]"
              << " transform_rank=" << t.size() << " batch_rank=" << blist.size()
              << " threads=" << plan.nthreads << " flags="
              << (plan.flags == FFTW_ESTIMATE  ? "ESTIMATE"
                  : plan.flags == FFTW_MEASURE ? "MEASURE"
                  : plan.flags == FFTW_PATIENT ? "PATIENT"
                                               : "EXHAUSTIVE")
              << "\n";
  }

  return 0;
}

int fftndExecute(FFTNDHandle plan, void*& idata, void*& odata, shafft::FFTDirection direction) {
  if (plan.nsubplans == 0) {
    if (diagEnabled()) {
      dbgHdr(plan.comm, "fftndExecute");
      std::cerr << "no-op (nsubplans=0)\n";
    }
    return 0;
  }

  if (diagEnabled()) {
    dbgHdr(plan.comm, "fftndExecute");
    std::cerr << (direction == shafft::FFTDirection::FORWARD ? "FORWARD" : "BACKWARD")
              << " idata=" << idata << " odata=" << odata << "\n";
  }

  if (plan.fft_type == shafft::FFTType::Z2Z) {
    if (direction == shafft::FFTDirection::FORWARD)
      fftw_execute_dft(plan.fwd_dp,
                       reinterpret_cast<fftw_complex*>(idata),
                       reinterpret_cast<fftw_complex*>(odata));
    else
      fftw_execute_dft(plan.bwd_dp,
                       reinterpret_cast<fftw_complex*>(idata),
                       reinterpret_cast<fftw_complex*>(odata));
  } else {
    if (direction == shafft::FFTDirection::FORWARD)
      fftwf_execute_dft(plan.fwd_sp,
                        reinterpret_cast<fftwf_complex*>(idata),
                        reinterpret_cast<fftwf_complex*>(odata));
    else
      fftwf_execute_dft(plan.bwd_sp,
                        reinterpret_cast<fftwf_complex*>(idata),
                        reinterpret_cast<fftwf_complex*>(odata));
  }

  // FFTW always does idata->odata (nsubplans=1), swap so result is in idata
  if (plan.nsubplans % 2 == 1) {
    std::swap(idata, odata);
  }
  return 0;
}

int fftndDestroy(FFTNDHandle plan) {
  // Clean up plans (null-safe via checks)
  if (plan.fwd_dp)
    fftw_destroy_plan(plan.fwd_dp);
  if (plan.bwd_dp)
    fftw_destroy_plan(plan.bwd_dp);
  if (plan.fwd_sp)
    fftwf_destroy_plan(plan.fwd_sp);
  if (plan.bwd_sp)
    fftwf_destroy_plan(plan.bwd_sp);

  // Note: We do NOT call fftw_cleanup_threads() or fftwf_cleanup_threads() here
  // because other plans may still be active (multiple subplans are destroyed in
  // a loop, and other FFTND or FFT1D instances may exist). Call fftndFinalize()
  // at program exit instead.

  return 0;
}

int fftndFinalize() noexcept {
  fftw_cleanup_threads();
  fftwf_cleanup_threads();
  return 0;
}
