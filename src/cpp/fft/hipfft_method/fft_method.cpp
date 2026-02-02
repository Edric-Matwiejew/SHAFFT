// fft_method.cpp  (hipFFT back-end)
//
// This module handles decomposition of N-dimensional FFTs into batched <=3D FFTs
// that hipFFT can execute. Uses hipfftMakePlanMany64 for 64-bit stride/dist/batch
// support, avoiding 32-bit overflow issues with large tensors.
//
// Key design notes:
// 1. hipFFT only supports up to 3D transforms natively, so higher dimensions
//    are decomposed into multiple batched subplans.
//
// 2. We use hipfftMakePlanMany64 which accepts long long parameters for
//    strides, distances, and batch counts, avoiding 32-bit overflow.
//
// 3. For non-trailing axis blocks with many superbatches (leading dimensions),
//    we use gpuTT to transpose the data so that FFT axes become trailing,
//    perform the FFT without superbatch iteration, then transpose back.
//    This avoids precision degradation from repeated hipfftExec calls.
//
#include "fft_method.h"

#include <shafft/shafft_error.hpp>

#include "hipcheck.h"
#include "transpose_fft.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdlib>
#include <gputt.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mpi.h>  // for rank-tagging in diagnostics
#include <numeric>
#include <vector>

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
  if (!init)
    return -1;
  int r = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  return r;
}

static void dbg_hdr(const char* where) {
  if (!diag_enabled())
    return;
  int r = mpi_rank();
  std::cerr << "[SHAFFT:hipfft]"
            << (r >= 0 ? ("[r" + std::to_string(r) + "]") : std::string("[r?]")) << "[" << where
            << "] ";
}

static inline long long prod_range_ll(const int* a, int i0, int i1) {
  long long p = 1;
  if (i0 > i1)
    return 1;
  for (int i = i0; i <= i1; ++i)
    p *= static_cast<long long>(a[i]);
  return p;
}

template <typename T>
static void print_array(const char* name, const T* a, int n) {
  if (!diag_enabled())
    return;
  std::cerr << name << "=[";
  for (int i = 0; i < n; ++i) {
    if (i)
      std::cerr << ",";
    std::cerr << a[i];
  }
  std::cerr << "]";
}

// Decompose rank nta into {3,3,2,1,...} with fastest-first groups
static void transform_subsequence(int nta, int* nsubplans, int* subseq) {
  int seq3 = nta / 3;
  int rem = nta - 3 * seq3;
  int seq2 = rem / 2;
  int seq1 = rem - 2 * seq2;

  int idx = 0;
  for (int i = 0; i < seq3; ++i)
    subseq[idx++] = 3;
  for (int i = 0; i < seq2; ++i)
    subseq[idx++] = 2;
  for (int i = 0; i < seq1; ++i)
    subseq[idx++] = 1;
  *nsubplans = idx;
}

// Compute hipFFT parameters for a contiguous axis block that may be non-trailing.
// Fills n[] in SLOWEST-first order as required by hipFFT/cuFFT/FFTW convention:
//   n[0] = size[minAxis] (slowest), ..., n[nta-1] = size[maxAxis] (fastest).
// All output parameters are 64-bit for use with hipfftMakePlanMany64.
//
// For non-trailing blocks: we use batch=T (trailing product) with idist=1 to
// process all trailing positions in a single hipfftExec call. This avoids
// rocFFT issues with multiple hipfftExec calls using different input offsets.
//
// Returns 0 on success, non-zero on error.
static int transform_parameters_64(int ndim, const int size[], int nta, const int ta[],
                                   long long* n, long long* idist, long long* odist,
                                   long long* istride, long long* ostride, long long* batch,
                                   long long* superbatch, long long* superbatch_offset,
                                   long long* innerbatch, long long* innerbatch_offset) {
  // Locate the contiguous span [minAxis..maxAxis]
  std::vector<int> axes_sorted(ta, ta + nta);
  std::sort(axes_sorted.begin(), axes_sorted.end());
  for (int i = 0; i + 1 < nta; ++i) {
    if (axes_sorted[i] + 1 != axes_sorted[i + 1]) {
      fprintf(stderr, "Axes must be consecutive (got %d then %d)\n", axes_sorted[i],
              axes_sorted[i + 1]);
      return ERR_INVALID_DIM;
    }
  }

  const int minAxis = axes_sorted.front();
  const int maxAxis = axes_sorted.back();

  // Populate n[] in SLOWEST-to-FASTEST order for hipFFT convention:
  //   n[0] = size[minAxis], n[1] = size[minAxis+1], ..., n[nta-1] = size[maxAxis]
  // This ensures hipFFT computes correct strides for the physical memory layout.
  for (int i = 0; i < nta; ++i) {
    const int ax = minAxis + i;
    int dim = size[ax];
    if (dim <= 0) {
      fprintf(stderr, "Invalid dimension: dim[%d]=%d\n", ax, dim);
      return ERR_INVALID_DIM;
    }
    n[i] = static_cast<long long>(dim);
  }

  // trailing product beyond the block -> element stride for the fastest dim
  long long T = prod_range_ll(size, maxAxis + 1, ndim - 1);
  *istride = T;
  *ostride = T;

  if (maxAxis == ndim - 1) {
    // ===== Trailing block =====
    // FFT is on the trailing axes - batches are in the leading dimensions
    long long idist_ll = prod_range_ll(size, minAxis, ndim - 1);
    *idist = idist_ll;
    *odist = idist_ll;

    long long total_ll = prod_range_ll(size, 0, ndim - 1);
    *batch = total_ll / idist_ll;
    *superbatch = 1;
    *superbatch_offset = total_ll;  // Not used when superbatch=1

    // No inner loop needed for trailing blocks
    *innerbatch = 1;
    *innerbatch_offset = 0;  // Not used when innerbatch=1
  } else {
    // ===== Non-trailing block =====
    // Use batch=T (trailing product) with idist=1 to process all trailing
    // positions in a single hipfftExec call.
    // This avoids a rocFFT bug where multiple hipfftExec calls with different
    // input offsets produce incorrect results.
    *idist = 1;  // Distance between consecutive batch elements
    *odist = 1;
    *batch = T;  // Process all T trailing positions in one call

    // No inner loop needed - handled by batch
    *innerbatch = 1;
    *innerbatch_offset = 0;  // Not used when innerbatch=1

    // Outer loop iterates over leading dimensions (L positions)
    // The offset between superbatches is the stride of minAxis (product of dims from minAxis to
    // end)
    long long L = (minAxis > 0) ? prod_range_ll(size, 0, minAxis - 1) : 1;
    *superbatch = L;
    *superbatch_offset = prod_range_ll(size, minAxis, ndim - 1);  // stride of axis minAxis
  }
  return ERR_SUCCESS;
}

// ---------------- API ----------------

int fftPlan(fftHandle& plan, int nta, int* ta, int ndim, int* dimensions,
            shafft::FFTType precision) {
  switch (precision) {
    case shafft::FFTType::C2C:
      plan.fft_type = HIPFFT_C2C;
      break;
    case shafft::FFTType::Z2Z:
      plan.fft_type = HIPFFT_Z2Z;
      break;
    default:
      fprintf(stderr, "Invalid FFT type\n");
      return static_cast<int>(shafft::Status::SHAFFT_ERR_INVALID_FFTTYPE);
  }

  for (int i = 0; i < ndim; ++i) {
    if (dimensions[i] <= 0) {
      fprintf(stderr, "Invalid dimension[%d]=%d for hipFFT\n", i, dimensions[i]);
      return ERR_INVALID_DIM;
    }
  }

  if (diag_enabled()) {
    dbg_hdr("fftPlan");
    std::cerr << "ndim=" << ndim
              << " precision=" << (plan.fft_type == HIPFFT_Z2Z ? "Z2Z(double)" : "C2C(float)")
              << " ";
    print_array("dimensions", dimensions, ndim);
    std::cerr << " nta=" << nta << " ";
    print_array("ta", ta, nta);
    std::cerr << "\n";
  }

  // Decompose into <=3D chunks
  std::vector<int> subseq(std::max(nta, 1));
  int nsubplans_init = 0;
  transform_subsequence(nta, &nsubplans_init, subseq.data());
  subseq.resize(nsubplans_init);

  plan.nsubplans = static_cast<int>(subseq.size());

  if (diag_enabled()) {
    dbg_hdr("fftPlan");
    std::cerr << "subsequence=";
    print_array("", subseq.data(), plan.nsubplans);
    std::cerr << "\n";
  }

  // Allocate per-subplan state
  plan.subplans = new hipfftHandle[plan.nsubplans]();
  plan.superbatches = new long long[plan.nsubplans];
  plan.superbatches_offset = new long long[plan.nsubplans];
  plan.innerbatches = new long long[plan.nsubplans];
  plan.innerbatches_offset = new long long[plan.nsubplans];
  plan.transpose_info = new TransposeFFTInfo[plan.nsubplans]();

  const long long sb_threshold = get_superbatch_threshold();
  int ta_taken = 0;

  // Create each subplan
  for (int j = 0; j < plan.nsubplans; ++j) {
    int r = subseq[j];
    assert(r >= 1 && r <= 3);
    ta_taken += r;

    const int* ta_win = &ta[nta - ta_taken];

    std::vector<long long> n_j(r);
    long long idist = 0, odist = 0, istride = 0, ostride = 0, batch = 0;
    long long superbatch = 0, superbatch_offset = 0;
    long long innerbatch = 0, innerbatch_offset = 0;

    int tp_rc = transform_parameters_64(ndim, dimensions, r, ta_win, n_j.data(), &idist, &odist,
                                        &istride, &ostride, &batch, &superbatch, &superbatch_offset,
                                        &innerbatch, &innerbatch_offset);
    if (tp_rc != ERR_SUCCESS)
      return tp_rc;

    if (diag_enabled()) {
      dbg_hdr("fftPlan");
      std::cerr << "subplan j=" << j << " rank=" << r << " ta_win=";
      print_array("", ta_win, r);
      std::cerr << " n=";
      print_array("", n_j.data(), r);
      std::cerr << " istride=" << istride << " idist=" << idist << " batch=" << batch
                << " superbatch=" << superbatch << " superbatch_offset=" << superbatch_offset
                << " innerbatch=" << innerbatch << " innerbatch_offset=" << innerbatch_offset
                << "\n";
    }

    plan.superbatches[j] = superbatch;
    plan.superbatches_offset[j] = superbatch_offset;
    plan.innerbatches[j] = innerbatch;
    plan.innerbatches_offset[j] = innerbatch_offset;

    // Check if we should use transpose-based FFT for this subplan
    if (superbatch > sb_threshold) {
      if (diag_enabled()) {
        dbg_hdr("fftPlan");
        std::cerr << "  superbatch=" << superbatch << " exceeds threshold=" << sb_threshold
                  << ", creating transpose-based FFT\n";
      }

      int rc = create_transpose_fft(ndim, dimensions, r, ta_win, plan.fft_type, plan.stream,
                                    plan.transpose_info[j]);
      if (rc != ERR_SUCCESS)
        return rc;

      if (plan.transpose_info[j].enabled) {
        // Transpose-based FFT successfully created, skip regular plan creation
        plan.subplans[j] = nullptr;
        if (diag_enabled()) {
          dbg_hdr("fftPlan");
          std::cerr << "  transpose-based FFT enabled for subplan " << j << "\n";
        }
        continue;
      }
      // If transpose creation returned success but info.enabled is false,
      // fall through to regular plan creation (e.g., already trailing axes)
    }

    if (int rc = hipfftCheck(hipfftCreate(&plan.subplans[j])))
      return rc;
    if (int rc = hipfftCheck(hipfftSetStream(plan.subplans[j], plan.stream)))
      return rc;

    size_t worksize = 0;
    // embed = n for contiguous layouts (hipFFT convention)
    std::vector<long long> inembed(n_j);
    std::vector<long long> onembed(n_j);

    // Use hipfftMakePlanMany64 for 64-bit parameter support
    if (int rc = hipfftCheck(hipfftMakePlanMany64(plan.subplans[j], r, n_j.data(), inembed.data(),
                                                  istride, idist, onembed.data(), ostride, odist,
                                                  plan.fft_type, batch, &worksize)))
      return rc;

    if (diag_enabled()) {
      dbg_hdr("fftPlan");
      std::cerr << "  created plan: worksize=" << worksize << "\n";
    }
  }
  return 0;
}

int fftSetStream(fftHandle& plan, hipStream_t stream) {
  plan.stream = stream;
  for (int j = 0; j < plan.nsubplans; ++j) {
    // Set stream on transpose-based FFT if enabled
    if (plan.transpose_info && plan.transpose_info[j].enabled) {
      if (int rc = hipfftCheck(hipfftSetStream(plan.transpose_info[j].transposed_fft, stream)))
        return rc;
      continue;
    }
    // Regular subplan
    if (plan.subplans[j] != nullptr) {
      if (int rc = hipfftCheck(hipfftSetStream(plan.subplans[j], stream)))
        return rc;
    }
  }
  return 0;
}

int fftExecute(fftHandle plan, void*& idata, void*& odata, shafft::FFTDirection direction) {
  int dir = (direction == shafft::FFTDirection::FORWARD) ? HIPFFT_FORWARD : HIPFFT_BACKWARD;

  // Buffer management: track which buffer currently holds the data.
  // - buf[0] = idata (first buffer passed in)
  // - buf[1] = odata (second buffer passed in)
  // - data_in_buf0: true if current data is in buf[0], false if in buf[1]
  //
  // Both regular and transpose paths flip the buffer each subplan:
  // - Regular path: reads from src, writes to dst -> flips
  // - Transpose path: src->dst (transpose) -> dst->src (FFT) -> src->dst (transpose back) -> flips
  //
  // Result location is predictable: buf[1] if nsubplans odd, buf[0] if even.
  void* buf[2] = {idata, odata};
  bool data_in_buf0 = true;  // Input starts in buf[0] (idata)

  if (diag_enabled()) {
    dbg_hdr("fftExecute");
    std::cerr << "begin " << (direction == shafft::FFTDirection::FORWARD ? "FORWARD" : "BACKWARD")
              << " nsubplans=" << plan.nsubplans << " buf[0]=" << buf[0] << " buf[1]=" << buf[1]
              << "\n";
  }

  for (int j = 0; j < plan.nsubplans; ++j) {
    // Determine source (where data currently is) and destination (other buffer)
    void* src = data_in_buf0 ? buf[0] : buf[1];
    void* dst = data_in_buf0 ? buf[1] : buf[0];

    // Check for transpose-based FFT execution
    if (plan.transpose_info && plan.transpose_info[j].enabled) {
      TransposeFFTInfo& ti = plan.transpose_info[j];

      if (diag_enabled()) {
        dbg_hdr("fftExecute");
        std::cerr << "subplan j=" << j
                  << " (transpose) data_in=" << (data_in_buf0 ? "buf[0]" : "buf[1]") << "\n";
      }

      // Transpose path (consistent flip behavior - data goes from src to dst):
      // 1. Transpose src -> dst (data now in dst, transposed layout)
      // 2. FFT out-of-place dst -> src (data now in src, still transposed)
      // 3. Transpose src -> dst (data now in dst, original layout)
      // Net effect: data moves from src to dst (same as regular path - buffer flip!)

      // Step 1: Transpose src -> dst
      gputtResult gpuRes = gputtExecute(ti.transpose_to_front, src, dst);
      if (gpuRes != GPUTT_SUCCESS) {
        fprintf(stderr, "gputtExecute (to_front) failed for subplan %d with error %d\n", j, gpuRes);
        return ERR_RUNTIME;
      }

      // Step 2: FFT out-of-place dst -> src
      switch (plan.fft_type) {
        case HIPFFT_C2C:
          if (int rc = hipfftCheck(hipfftExecC2C(ti.transposed_fft,
                                                 reinterpret_cast<hipfftComplex*>(dst),
                                                 reinterpret_cast<hipfftComplex*>(src), dir)))
            return rc;
          break;
        case HIPFFT_Z2Z:
          if (int rc = hipfftCheck(hipfftExecZ2Z(ti.transposed_fft,
                                                 reinterpret_cast<hipfftDoubleComplex*>(dst),
                                                 reinterpret_cast<hipfftDoubleComplex*>(src), dir)))
            return rc;
          break;
        default:
          assert(false && "Unsupported fft_type");
      }

      // Step 3: Transpose back src -> dst (data ends up in dst with original layout)
      gpuRes = gputtExecute(ti.transpose_to_back, src, dst);
      if (gpuRes != GPUTT_SUCCESS) {
        fprintf(stderr, "gputtExecute (to_back) failed for subplan %d with error %d\n", j, gpuRes);
        return ERR_RUNTIME;
      }

      // Data is now in dst - buffer flip occurred, same as regular path
      data_in_buf0 = !data_in_buf0;

      if (diag_enabled()) {
        dbg_hdr("fftExecute");
        std::cerr << "  transpose complete, data now in " << (data_in_buf0 ? "buf[0]" : "buf[1]")
                  << "\n";
      }
      continue;
    }

    // Regular (non-transpose) execution path: src -> dst
    const long long sb_count = plan.superbatches[j];
    const long long sb_offset = plan.superbatches_offset[j];
    const long long ib_count = plan.innerbatches[j];
    const long long ib_offset = plan.innerbatches_offset[j];

    if (sb_count <= 0 || sb_offset <= 0) {
      fprintf(stderr, "Invalid superbatch parameters at execution time.\n");
      return ERR_RUNTIME;
    }

    if (diag_enabled()) {
      dbg_hdr("fftExecute");
      std::cerr << "subplan j=" << j << " (regular)" << " sb=" << sb_count << "x" << sb_offset
                << " ib=" << ib_count << "x" << ib_offset << " src=" << src << " dst=" << dst
                << "\n";
    }

    // Execute batched FFTs with superbatch iteration
    for (long long k = 0; k < sb_count; ++k) {
      const size_t sb_off = static_cast<size_t>(k) * static_cast<size_t>(sb_offset);

      for (long long m = 0; m < ib_count; ++m) {
        const size_t off = sb_off + static_cast<size_t>(m) * static_cast<size_t>(ib_offset);

        switch (plan.fft_type) {
          case HIPFFT_C2C:
            if (int rc = hipfftCheck(
                    hipfftExecC2C(plan.subplans[j], &reinterpret_cast<hipfftComplex*>(src)[off],
                                  &reinterpret_cast<hipfftComplex*>(dst)[off], dir)))
              return rc;
            break;
          case HIPFFT_Z2Z:
            if (int rc = hipfftCheck(hipfftExecZ2Z(
                    plan.subplans[j], &reinterpret_cast<hipfftDoubleComplex*>(src)[off],
                    &reinterpret_cast<hipfftDoubleComplex*>(dst)[off], dir)))
              return rc;
            break;
          default:
            assert(false && "Unsupported fft_type");
        }
      }
    }

    // Data is now in dst
    data_in_buf0 = !data_in_buf0;

    if (diag_enabled()) {
      dbg_hdr("fftExecute");
      std::cerr << "  complete, data now in " << (data_in_buf0 ? "buf[0]" : "buf[1]") << "\n";
    }
  }

  // Both regular and transpose paths now flip the buffer consistently.
  // Result location is predictable based on nsubplans parity:
  // - nsubplans odd  -> data in buf[1] (odata)
  // - nsubplans even -> data in buf[0] (idata)
  // We swap the caller's pointers so result is always accessible via idata.

  if (int rc = hipCheck(hipStreamSynchronize(plan.stream)))
    return rc;

  if (plan.nsubplans % 2 == 1) {
    // Result is in odata, swap so caller finds it in idata
    std::swap(idata, odata);
  }

  if (diag_enabled()) {
    dbg_hdr("fftExecute");
    std::cerr << "end, result in idata=" << idata << "\n";
  }
  return 0;
}

int fftDestroy(fftHandle plan) {
  int first_error = 0;

  // Destroy transpose-based FFT resources
  if (plan.transpose_info) {
    for (int j = 0; j < plan.nsubplans; ++j) {
      destroy_transpose_fft(plan.transpose_info[j]);
    }
    delete[] plan.transpose_info;
    plan.transpose_info = nullptr;
  }

  // Destroy regular hipFFT subplans
  if (plan.subplans) {
    for (int j = 0; j < plan.nsubplans; ++j) {
      if (!plan.subplans[j])
        continue;
      hipfftResult_t res = hipfftDestroy(plan.subplans[j]);
      if (res != HIPFFT_SUCCESS && first_error == 0) {
        first_error = static_cast<int>(res);
        fprintf(stderr, "hipfftDestroy failed for subplan %d: %d at %s:%d\n", j, first_error,
                __FILE__, __LINE__);
      }
    }
    delete[] plan.subplans;
  }
  delete[] plan.superbatches;
  delete[] plan.superbatches_offset;
  delete[] plan.innerbatches;
  delete[] plan.innerbatches_offset;
  plan.subplans = nullptr;
  plan.superbatches = nullptr;
  plan.superbatches_offset = nullptr;
  plan.innerbatches = nullptr;
  plan.innerbatches_offset = nullptr;
  plan.nsubplans = 0;
  return first_error;
}
