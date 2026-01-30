// fft_method_fftw.cpp  (FFTW3 CPU back-end using guru64 interface)
// Build with -DFFTW3 and link with -lfftw3 -lfftw3f -lfftw3_threads -lfftw3f_threads.
//
// Environment variables:
//   SHAFFT_FFTW_THREADS  - Number of threads for FFTW (default: 1)
//   SHAFFT_FFTW_PLANNER  - Planner strategy: ESTIMATE (default), MEASURE, PATIENT, EXHAUSTIVE

#include "fft_method.h"          // declares fftPlan / fftExecute / fftDestroy
#include <shafft/shafft_types.hpp>      // shafft::FFTType, shafft::FFTDirection
#include <fftw3.h>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------- helpers ----------------

static int getenv_threads_or_default(const char* key, int defv = 1) {
  if (const char* s = std::getenv(key)) {
    int v = std::atoi(s);
    return v > 0 ? v : defv;
  }
  return defv;
}

// Parse SHAFFT_FFTW_PLANNER env var: ESTIMATE (default), MEASURE, PATIENT, EXHAUSTIVE
static unsigned getenv_planner_flags() {
  const char* s = std::getenv("SHAFFT_FFTW_PLANNER");
  if (!s) return FFTW_ESTIMATE;
  std::string val(s);
  // Convert to uppercase for comparison
  for (char& c : val) c = std::toupper(static_cast<unsigned char>(c));
  if (val == "MEASURE")    return FFTW_MEASURE;
  if (val == "PATIENT")    return FFTW_PATIENT;
  if (val == "EXHAUSTIVE") return FFTW_EXHAUSTIVE;
  return FFTW_ESTIMATE;  // default
}

// Compute total elements from dimensions array
static size_t compute_total_elements(int ndim, const int* dims) {
  size_t n = 1;
  for (int i = 0; i < ndim; ++i) n *= (size_t)dims[i];
  return n;
}

// Row-major element stride for axis m is product of sizes to its right: prod(dims[m+1..])
static void build_guru64_dims(int ndim, const int* dims,
                              int a, int b,
                              std::vector<fftw_iodim64>& t,
                              std::vector<fftw_iodim64>& blist)
{
  std::vector<long long> P(ndim + 1, 1);
  for (int m = ndim - 1; m >= 0; --m) P[m] = P[m + 1] * (long long)dims[m];

  // Transform dims: contiguous block [a..b]
  const int r = b - a + 1;
  t.resize(r);
  for (int k = 0; k < r; ++k) {
    const int m = a + k;
    t[k].n  = (ptrdiff_t)dims[m];
    t[k].is = (ptrdiff_t)P[m + 1];
    t[k].os = (ptrdiff_t)P[m + 1];
  }

  // Batch dims: all axes except [a..b]
  blist.clear();
  blist.reserve(ndim - r);
  for (int m = 0; m < ndim; ++m) {
    if (m >= a && m <= b) continue;
    fftw_iodim64 d;
    d.n  = (ptrdiff_t)dims[m];
    d.is = (ptrdiff_t)P[m + 1];
    d.os = (ptrdiff_t)P[m + 1];
    blist.push_back(d);
  }
}

// ---------------- API ----------------

int fftPlan(fftHandle& plan,
            int nta, int* ta,
            int ndim, int* dimensions,
            shafft::FFTType precision)
{
  plan.fft_type = precision;
  plan.ndim = ndim;
  plan.nta  = nta;

  if (ndim <= 0) throw std::invalid_argument("FFTW: ndim must be > 0");
  for (int i = 0; i < ndim; ++i)
    if (dimensions[i] <= 0) throw std::invalid_argument("FFTW: dimensions must be > 0");

  // Handle degenerate stage (no local axes to transform)
  if (nta == 0) {
    plan.nsubplans = 0; // upstream will detect even parity -> no swap
    return 0;
  }

  // Find contiguous block [a..b] from ta[]
  int a = ta[0], b = ta[0];
  for (int i = 1; i < nta; ++i) { a = std::min(a, ta[i]); b = std::max(b, ta[i]); }
  if (b - a + 1 != nta) throw std::invalid_argument("FFTW: transform axes must form a contiguous block");
  plan.a = a; plan.b = b;

  // Build guru64 dims
  std::vector<fftw_iodim64> t, blist;
  build_guru64_dims(ndim, dimensions, a, b, t, blist);

  // Threads: initialise threading for the precision being used
  plan.nthreads = getenv_threads_or_default("SHAFFT_FFTW_THREADS", 1);
  if (precision == shafft::FFTType::Z2Z) {
    fftw_init_threads();
    fftw_plan_with_nthreads(plan.nthreads);
  } else {
    fftwf_init_threads();
    fftwf_plan_with_nthreads(plan.nthreads);
  }

  // Plan flags from environment variable
  plan.flags = getenv_planner_flags();

  // Allocate properly-sized dummy buffers for planning (needed for FFTW_MEASURE etc.)
  const size_t total_elems = compute_total_elements(ndim, dimensions);
  fftw_complex*  dummy_dp  = nullptr;
  fftwf_complex* dummy_sp  = nullptr;

  if (precision == shafft::FFTType::Z2Z) {
    dummy_dp = (fftw_complex*) fftw_malloc(total_elems * sizeof(fftw_complex));
    if (!dummy_dp) throw std::runtime_error("FFTW: dummy allocation failed (double)");
  } else {
    dummy_sp = (fftwf_complex*) fftwf_malloc(total_elems * sizeof(fftwf_complex));
    if (!dummy_sp) throw std::runtime_error("FFTW: dummy allocation failed (float)");
  }

  if (precision == shafft::FFTType::Z2Z) {
    plan.fwd_dp = fftw_plan_guru64_dft((int)t.size(), t.data(),
                                       (int)blist.size(), blist.data(),
                                       dummy_dp, dummy_dp,
                                       FFTW_FORWARD, plan.flags);
    plan.bwd_dp = fftw_plan_guru64_dft((int)t.size(), t.data(),
                                       (int)blist.size(), blist.data(),
                                       dummy_dp, dummy_dp,
                                       FFTW_BACKWARD, plan.flags);
    fftw_free(dummy_dp);
    if (!plan.fwd_dp || !plan.bwd_dp)
      throw std::runtime_error("FFTW: plan creation failed (double)");
  } else { // single precision
    std::vector<fftwf_iodim64> tf(t.size()), bf(blist.size());
    for (size_t i = 0; i < t.size(); ++i) { tf[i].n = t[i].n; tf[i].is = t[i].is; tf[i].os = t[i].os; }
    for (size_t i = 0; i < blist.size(); ++i) { bf[i].n = blist[i].n; bf[i].is = blist[i].is; bf[i].os = blist[i].os; }

    plan.fwd_sp = fftwf_plan_guru64_dft((int)tf.size(), tf.data(),
                                        (int)bf.size(), bf.data(),
                                        dummy_sp, dummy_sp,
                                        FFTW_FORWARD, plan.flags);
    plan.bwd_sp = fftwf_plan_guru64_dft((int)tf.size(), tf.data(),
                                        (int)bf.size(), bf.data(),
                                        dummy_sp, dummy_sp,
                                        FFTW_BACKWARD, plan.flags);
    fftwf_free(dummy_sp);
    if (!plan.fwd_sp || !plan.bwd_sp)
      throw std::runtime_error("FFTW: plan creation failed (single)");
  }

  plan.nsubplans = 1;  // odd so upstream swaps buffers once

  return 0;
}

int fftExecute(fftHandle plan, void* idata, void* odata,
               shafft::FFTDirection direction)
{
  if (plan.nsubplans == 0) return 0; // no-op stage

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
  return 0;
}

int fftDestroy(fftHandle plan)
{
  // Clean up plans and threading state for the precision that was used
  if (plan.fwd_dp || plan.bwd_dp) {
    if (plan.fwd_dp) fftw_destroy_plan(plan.fwd_dp);
    if (plan.bwd_dp) fftw_destroy_plan(plan.bwd_dp);
    fftw_cleanup_threads();
  }
  if (plan.fwd_sp || plan.bwd_sp) {
    if (plan.fwd_sp) fftwf_destroy_plan(plan.fwd_sp);
    if (plan.bwd_sp) fftwf_destroy_plan(plan.bwd_sp);
    fftwf_cleanup_threads();
  }
  return 0;
}
