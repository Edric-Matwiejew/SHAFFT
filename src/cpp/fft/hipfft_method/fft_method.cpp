// fft_method.cpp  (hipFFT back-end)
#include "fft_method.h"
#include "hipcheck.h"
#include <shafft/shafft_error.hpp>

#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>
#include <limits>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <mpi.h>   // for rank-tagging in diagnostics

static constexpr int ERR_SUCCESS = static_cast<int>(shafft::Status::SHAFFT_SUCCESS);
static constexpr int ERR_INVALID_DIM = static_cast<int>(shafft::Status::SHAFFT_ERR_INVALID_DIM);
static constexpr int ERR_RUNTIME = static_cast<int>(shafft::Status::SHAFFT_ERR_BACKEND);

// ---------------- diagnostics toggles ----------------
#ifndef SHAFFT_FFT_DIAG
#define SHAFFT_FFT_DIAG 0
#endif

static inline bool diag_enabled() {
#if SHAFFT_FFT_DIAG
  return true;
#else
  static int cached = -1;
  if (cached < 0) {
    const char* env = std::getenv("SHAFFT_FFT_DIAG");
    cached = (env && env[0] == '1') ? 1 : 0;
  }
  return cached != 0;
#endif
}

static inline int mpi_rank() {
  int init = 0;
  MPI_Initialized(&init);
  if (!init) return -1;
  int r = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  return r;
}

static void dbg_hdr(const char* where) {
  if (!diag_enabled()) return;
  int r = mpi_rank();
  std::cerr << "[SHAFFT:diag]"
            << (r >= 0 ? ("[r" + std::to_string(r) + "]") : std::string("[r?]"))
            << "[" << where << "] ";
}

static inline long long prod_range_ll(const int* a, int i0, int i1) {
  long long p = 1;
  if (i0 > i1) return 1;
  for (int i = i0; i <= i1; ++i) p *= static_cast<long long>(a[i]);
  return p;
}

static void print_array(const char* name, const int* a, int n) {
  if (!diag_enabled()) return;
  std::cerr << name << "=[";
  for (int i = 0; i < n; ++i) {
    if (i) std::cerr << ",";
    std::cerr << a[i];
  }
  std::cerr << "]";
}

// Return stride threshold (elements) for non-trailing advanced-stride subplans.
// Default = 20000 elements; override with env SHAFFT_MAX_STRIDE_ELEMS.
static inline int stride_thresh_elems() {
  static int v = -1;
  if (v >= 0) return v;
  const char* env = std::getenv("SHAFFT_MAX_STRIDE_ELEMS");
  if (env && *env) {
    long long tmp = std::atoll(env);
    if (tmp < 1) tmp = 1;
    if (tmp > std::numeric_limits<int>::max()) tmp = std::numeric_limits<int>::max();
    v = static_cast<int>(tmp);
  } else {
    v=20000;
  }
  return v;
}

// Decompose rank nta into {3,3,2,1,...} with fastest-first groups
static void transform_subsequence(int nta, int* nsubplans, int* subseq) {
  int seq3 = nta / 3;
  int rem  = nta - 3 * seq3;
  int seq2 = rem / 2;
  int seq1 = rem - 2 * seq2;

  int idx = 0;
  for (int i = 0; i < seq3; ++i) subseq[idx++] = 3;
  for (int i = 0; i < seq2; ++i) subseq[idx++] = 2;
  for (int i = 0; i < seq1; ++i) subseq[idx++] = 1;
  *nsubplans = idx;
}

// Compute hipFFT parameters for a contiguous axis block that may be non-trailing.
// Fills n[] in fastest-first order: n[0] = size[maxAxis], ..., n[nta-1] = size[minAxis].
// Returns 0 on success, non-zero on error.
static int transform_parameters(int ndim, const int size[],
                                 int nta, const int ta[],    // contiguous axes (any order)
                                 int* n,                     // length nta, fastest-first
                                 int* idist, int* odist,
                                 int* istride, int* ostride,
                                 int* batch, int* superbatch)
{
  // ---- locate the contiguous span [minAxis..maxAxis]
  int minAxis = ta[0], maxAxis = ta[0];
  for (int i = 1; i < nta; ++i) {
    minAxis = std::min(minAxis, ta[i]);
    maxAxis = std::max(maxAxis, ta[i]);
  }
  if (maxAxis - minAxis + 1 != nta) {
    fprintf(stderr, "Axes must be consecutive\n");
    return ERR_INVALID_DIM;
  }

  for (int i = 0; i < nta; ++i) n[i] = size[minAxis + i];

  // trailing product beyond the block -> element stride for the fastest dim
  long long T = prod_range_ll(size, maxAxis + 1, ndim - 1);   // empty -> 1
  if (T <= 0) T = 1;
  if (T > std::numeric_limits<int>::max()) {
    fprintf(stderr, "istride too large for hipFFT int parameters\n");
    return ERR_RUNTIME;
  }

  *istride = static_cast<int>(T);
  *ostride = *istride;

  if (maxAxis == ndim - 1) {
    // ===== Trailing block =====
    long long idist_ll = prod_range_ll(size, minAxis, ndim - 1);  // full block
    if (idist_ll > std::numeric_limits<int>::max()) {
      fprintf(stderr, "idist too large for hipFFT int parameters\n");
      return ERR_RUNTIME;
    }

    *idist = static_cast<int>(idist_ll);
    *odist = *idist;

    long long total_ll = 1;
    for (int i = 0; i < ndim; ++i) total_ll *= static_cast<long long>(size[i]);

    long long b_ll = total_ll / idist_ll;  // batches across leading dims
    if (b_ll <= 0 || b_ll > std::numeric_limits<int>::max()) {
      fprintf(stderr, "batch too large for hipFFT int parameters\n");
      return ERR_RUNTIME;
    }

    *batch      = static_cast<int>(b_ll);
    *superbatch = 1;
  } else {
    // ===== Non-trailing block =====
    *idist = 1;
    *odist = 1;

    // batch across all trailing positions:
    //   number of distinct trailing index combinations = T
    *batch = static_cast<int>(T);

    // superbatch across the leading dimensions (can be zero-length range -> 1)
    long long L = (minAxis > 0) ? prod_range_ll(size, 0, minAxis - 1) : 1;
    if (L <= 0 || L > std::numeric_limits<int>::max()) {
      fprintf(stderr, "superbatch too large for hipFFT int parameters\n");
      return ERR_RUNTIME;
    }
    *superbatch = static_cast<int>(L);
  }
  return ERR_SUCCESS;
}

// ---------------- API ----------------

int fftPlan(fftHandle& plan, int nta, int* ta,
            int ndim, int* dimensions,
            shafft::FFTType precision)
{
  switch (precision) {
    case shafft::FFTType::C2C: plan.fft_type = HIPFFT_C2C; break;
    case shafft::FFTType::Z2Z: plan.fft_type = HIPFFT_Z2Z; break;
    default:
      fprintf(stderr, "Invalid FFT type\n");
      return static_cast<int>(shafft::Status::SHAFFT_ERR_INVALID_FFTTYPE);
  }

  if (diag_enabled()) {
    dbg_hdr("fftPlan");
    std::cerr << "ndim=" << ndim << " ";
    print_array("dimensions", dimensions, ndim);
    std::cerr << " nta=" << nta << " ";
    print_array("ta", ta, nta);
    std::cerr << "\n";
  }

  // Start with default <=3D chunks.
  std::vector<int> subseq(std::max(nta, 1));
  int nsubplans_init = 0;
  transform_subsequence(nta, &nsubplans_init, subseq.data());
  subseq.resize(nsubplans_init);

  // ---- Prepass: stride-threshold aware splitting (no runtime probing) ----
  // We simulate subplans in order; if a non-trailing block has too-large istride,
  // split r -> (r-1, 1). This changes the subsequent ta windowing.
  int ta_taken_sim = 0;
  const int stride_limit = stride_thresh_elems();

  for (size_t j = 0; j < subseq.size(); ++j) {
    int r = subseq[j];
    assert(r >= 1 && r <= 3);
    // window into the tail of ta for this (possibly updated) r
    const int* ta_win = &ta[nta - (ta_taken_sim + r)];

    std::vector<int> n_j(r);
    int idist=0, odist=0, istride=0, ostride=0, batch=0, superbatch=0;
    transform_parameters(ndim, dimensions, r, ta_win,
                         n_j.data(), &idist, &odist, &istride, &ostride,
                         &batch, &superbatch);

    const bool is_non_trailing = (idist == 1 && odist == 1);
    if (is_non_trailing && istride > stride_limit && r > 1)
    {
      if (diag_enabled()) {
        dbg_hdr("fftPlan");
        std::cerr << "split subplan due to large stride: "
                  << "r=" << r << " istride=" << istride
                  << " limit=" << stride_limit
                  << " -> (" << (r-1) << ", 1)\n";
      }
      // Replace current rank with r-1 and insert a following rank-1.
      subseq[j] = r - 1;
      subseq.insert(subseq.begin() + j + 1, 1);
      // Do NOT advance ta_taken_sim yet; re-simulate this same j with r-1.
      --j;
      continue;
    }

    // Accept this subplan and advance the ta window by r
    ta_taken_sim += r;
  }

  // Now we know final number of subplans
  plan.nsubplans = static_cast<int>(subseq.size());
  if (diag_enabled()) {
    dbg_hdr("fftPlan");
    std::cerr << "final subsequence (after threshold splitting)=";
    print_array("", subseq.data(), plan.nsubplans);
    std::cerr << "\n";
  }

  // Allocate per-subplan state
  plan.subplans            = new hipfftHandle[plan.nsubplans];
  plan.superbatches        = new int[plan.nsubplans];
  plan.superbatches_offset = new int[plan.nsubplans];

  int ta_taken = 0;
  const long long total_ll =
      std::accumulate(dimensions, dimensions + ndim, 1LL, std::multiplies<long long>());

  if (total_ll <= 0) {
    fprintf(stderr, "Invalid total tensor size\n");
    return ERR_INVALID_DIM;
  }

  // Create each subplan using the finalized subsequence
  for (int j = 0; j < plan.nsubplans; ++j) {
    int r = subseq[j];
    assert(r >= 1 && r <= 3);
    ta_taken += r;

    const int* ta_win = &ta[nta - ta_taken];

    if (diag_enabled()) {
      dbg_hdr("fftPlan");
      std::cerr << "subplan j=" << j << " rank=" << r << " ta_win=";
      print_array("", ta_win, r);
      std::cerr << " total=" << total_ll << "\n";
    }

    std::vector<int> n_j(r);
    int idist=0, odist=0, istride=0, ostride=0, batch=0, superbatch=0;

    int tp_rc = transform_parameters(ndim, dimensions, r, ta_win,
                         n_j.data(), &idist, &odist, &istride, &ostride,
                         &batch, &superbatch);
    if (tp_rc != ERR_SUCCESS) return tp_rc;

    // (Optional) Informative warning if a rank-1 still exceeds threshold
    if (diag_enabled()) {
      const bool is_non_trailing = (idist == 1 && odist == 1);
      if (is_non_trailing && istride > stride_limit && r == 1) {
        dbg_hdr("fftPlan");
        std::cerr << "rank-1 subplan has istride=" << istride
                  << " exceeding limit=" << stride_limit
                  << " (cannot split further)\n";
      }
    }

    if (int rc = hipfftCheck(hipfftCreate(&plan.subplans[j]))) return rc;
    if (int rc = hipfftCheck(hipfftSetStream(plan.subplans[j], plan.stream))) return rc;

    if (superbatch <= 0 || (total_ll % superbatch) != 0) {
      fprintf(stderr, "Invalid superbatch partitioning for hipFFT plan.\n");
      return ERR_RUNTIME;
    }
    long long sb_offset_ll = total_ll / superbatch;
    if (sb_offset_ll > std::numeric_limits<int>::max()) {
      fprintf(stderr, "superbatch offset too large for hipFFT int parameters\n");
      return ERR_RUNTIME;
    }

    plan.superbatches[j]        = superbatch;
    plan.superbatches_offset[j] = static_cast<int>(sb_offset_ll);

    size_t worksize = 0;
    std::vector<int> inembed(n_j);
    std::vector<int> onembed(n_j);
    if (int rc = hipfftCheck(hipfftMakePlanMany(plan.subplans[j],
                                   r,
                                   n_j.data(),
                                   inembed.data(), /*istride*/ istride, /*idist*/ idist,
                                   onembed.data(), /*ostride*/ ostride, /*odist*/ odist,
                                   plan.fft_type,
                                   batch, &worksize))) return rc;

    if (diag_enabled()) {
      dbg_hdr("fftPlan");
      std::cerr << "j=" << j << " n=";
      print_array("", n_j.data(), r);
      std::cerr << " idist=" << idist
                << " odist=" << odist
                << " istride=" << istride
                << " ostride=" << ostride
                << " batch=" << batch
                << " superbatch=" << superbatch
                << " sb_offset=" << plan.superbatches_offset[j]
                << " worksize=" << worksize
                << "\n";
    }
  }
  return 0;
}

int fftSetStream(fftHandle& plan, hipStream_t stream) {
  plan.stream = stream;
  for (int j = 0; j < plan.nsubplans; ++j) {
    if (int rc = hipfftCheck(hipfftSetStream(plan.subplans[j], plan.stream))) return rc;
  }
  return 0;
}

int fftExecute(fftHandle plan, void* idata, void* odata,
               shafft::FFTDirection direction)
{
  int dir = (direction == shafft::FFTDirection::FORWARD) ? HIPFFT_FORWARD
                                                         : HIPFFT_BACKWARD;

  if (diag_enabled()) {
    dbg_hdr("fftExecute");
    std::cerr << "begin "
              << (direction == shafft::FFTDirection::FORWARD ? "FORWARD" : "BACKWARD")
              << " nsubplans=" << plan.nsubplans
              << " idata=" << idata << " odata=" << odata
              << "\n";
  }

  for (int j = 0; j < plan.nsubplans; ++j) {
    const int sb_count  = plan.superbatches[j];
    const int sb_offset = plan.superbatches_offset[j];

    if (sb_count <= 0 || sb_offset <= 0) {
      fprintf(stderr, "Invalid superbatch parameters at execution time.\\n");
      return ERR_RUNTIME;
    }

    if (diag_enabled()) {
      dbg_hdr("fftExecute");
      std::cerr << "subplan j=" << j
                << " sb_count=" << sb_count
                << " sb_offset=" << sb_offset
                << " idata_base=" << idata
                << " odata_base=" << odata
                << "\n";
    }

    for (int k = 0; k < sb_count; ++k) {
      if (diag_enabled()) {
        dbg_hdr("fftExecute");
        std::cerr << "  k=" << k
                  << " in=" << static_cast<void*>(
                         reinterpret_cast<hipfftDoubleComplex*>(idata) + (size_t)k * (size_t)sb_offset)
                  << " out=" << static_cast<void*>(
                         reinterpret_cast<hipfftDoubleComplex*>(odata) + (size_t)k * (size_t)sb_offset)
                  << "\n";
      }

      switch (plan.fft_type) {
        case HIPFFT_C2C:
          if (int rc = hipfftCheck(hipfftExecC2C(plan.subplans[j],
              &reinterpret_cast<hipfftComplex*>(idata)[(size_t)k * (size_t)sb_offset],
              &reinterpret_cast<hipfftComplex*>(odata)[(size_t)k * (size_t)sb_offset],
              dir))) return rc;
          break;
        case HIPFFT_Z2Z:
          if (int rc = hipfftCheck(hipfftExecZ2Z(plan.subplans[j],
              &reinterpret_cast<hipfftDoubleComplex*>(idata)[(size_t)k * (size_t)sb_offset],
              &reinterpret_cast<hipfftDoubleComplex*>(odata)[(size_t)k * (size_t)sb_offset],
              dir))) return rc;
          break;
        default:
          assert(false && "Unsupported fft_type");
      }
    }
    if (j < plan.nsubplans - 1) {
      if (diag_enabled()) { dbg_hdr("fftExecute"); std::cerr << "  swap(idata, odata)\n"; }
      std::swap(idata, odata);
    }
  }
  if (int rc = hipCheck(hipStreamSynchronize(plan.stream))) return rc;
  if (diag_enabled()) { dbg_hdr("fftExecute"); std::cerr << "end\n"; }
  return 0;
}

int fftDestroy(fftHandle plan) {
  int first_error = 0;
  for (int j = 0; j < plan.nsubplans; ++j) {
    hipfftResult_t res = hipfftDestroy(plan.subplans[j]);
    if (res != HIPFFT_SUCCESS && first_error == 0) {
      first_error = static_cast<int>(res);
      fprintf(stderr, "hipfftDestroy failed for subplan %d: %d at %s:%d\n",
              j, first_error, __FILE__, __LINE__);
    }
  }
  delete[] plan.subplans;
  delete[] plan.superbatches;
  delete[] plan.superbatches_offset;
  plan.subplans            = nullptr;
  plan.superbatches        = nullptr;
  plan.superbatches_offset = nullptr;
  plan.nsubplans           = 0;
  return first_error;
}

