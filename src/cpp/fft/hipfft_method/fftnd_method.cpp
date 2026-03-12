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
#include "fftnd_method.hpp"

#include <shafft/shafft_error.hpp>

#include "../../detail/array_utils.hpp"
#include "../common/diag.hpp"
#include "common/hipcheck.hpp"
#include "nd/transpose_fftnd.hpp"

#include <algorithm>
#include <cassert>
#include <gputt.h>
#include <iostream>
#include <mpi.h> // for rank-tagging in diagnostics
#include <vector>

static constexpr int errSuccess = static_cast<int>(shafft::Status::SUCCESS);
static constexpr int errInvalidDim = static_cast<int>(shafft::Status::ERR_INVALID_DIM);
static constexpr int errRuntime = static_cast<int>(shafft::Status::ERR_BACKEND);

using shafft::fft::diagEnabled;
using shafft::fft::mpiRank;
using shafft::fft::printArray;

// Module-specific debug header
static void dbgHdr(MPI_Comm comm, const char* where) {
  shafft::fft::dbgHdr("ND-hipfft", comm, where);
}

// Decompose rank nta into {3,3,2,1,...} with fastest-first groups
static void transformSubsequence(int nta, int* nsubplans, int* subseq) {
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
static int transformParameters64(int ndim,
                                 const int size[],
                                 int nta,
                                 const int ta[],
                                 long long* n,
                                 long long* idist,
                                 long long* odist,
                                 long long* istride,
                                 long long* ostride,
                                 long long* batch,
                                 long long* superbatch,
                                 long long* superbatchOffset,
                                 long long* innerbatch,
                                 long long* innerbatchOffset) {
  // Locate the contiguous span [minAxis..maxAxis]
  std::vector<int> axesSorted(ta, ta + nta);
  std::sort(axesSorted.begin(), axesSorted.end());
  for (int i = 0; i + 1 < nta; ++i) {
    if (axesSorted[i] + 1 != axesSorted[i + 1]) {
      fprintf(
          stderr, "Axes must be consecutive (got %d then %d)\n", axesSorted[i], axesSorted[i + 1]);
      return errInvalidDim;
    }
  }

  const int minAxis = axesSorted.front();
  const int maxAxis = axesSorted.back();

  // Populate n[] in SLOWEST-to-FASTEST order for hipFFT convention:
  //   n[0] = size[minAxis], n[1] = size[minAxis+1], ..., n[nta-1] = size[maxAxis]
  // This ensures hipFFT computes correct strides for the physical memory layout.
  for (int i = 0; i < nta; ++i) {
    const int ax = minAxis + i;
    int dim = size[ax];
    if (dim <= 0) {
      fprintf(stderr, "Invalid dimension: dim[%d]=%d\n", ax, dim);
      return errInvalidDim;
    }
    n[i] = static_cast<long long>(dim);
  }

  // trailing product beyond the block -> element stride for the fastest dim
  long long trailingProd = shafft::detail::prodRange<int, long long>(size, maxAxis + 1, ndim - 1);
  *istride = trailingProd;
  *ostride = trailingProd;

  if (maxAxis == ndim - 1) {
    // ===== Trailing block =====
    // FFT is on the trailing axes - batches are in the leading dimensions
    long long idistLl = shafft::detail::prodRange<int, long long>(size, minAxis, ndim - 1);
    *idist = idistLl;
    *odist = idistLl;

    long long totalLl = shafft::detail::prodRange<int, long long>(size, 0, ndim - 1);
    *batch = totalLl / idistLl;
    *superbatch = 1;
    *superbatchOffset = totalLl; // Not used when superbatch=1

    // No inner loop needed for trailing blocks
    *innerbatch = 1;
    *innerbatchOffset = 0; // Not used when innerbatch=1
  } else {
    // ===== Non-trailing block =====
    // Use batch=T (trailing product) with idist=1 to process all trailing
    // positions in a single hipfftExec call.
    // This avoids a rocFFT bug where multiple hipfftExec calls with different
    // input offsets produce incorrect results.
    *idist = 1; // Distance between consecutive batch elements
    *odist = 1;
    *batch = trailingProd; // Process all trailingProd trailing positions in one call

    // No inner loop needed - handled by batch
    *innerbatch = 1;
    *innerbatchOffset = 0; // Not used when innerbatch=1

    // Outer loop iterates over leading dimensions (leadingProd positions)
    // The offset between superbatches is the stride of minAxis (product of dims from minAxis to
    // end)
    long long leadingProd =
        (minAxis > 0) ? shafft::detail::prodRange<int, long long>(size, 0, minAxis - 1) : 1;
    *superbatch = leadingProd;
    *superbatchOffset = shafft::detail::prodRange<int, long long>(
        size, minAxis, ndim - 1); // stride of axis minAxis
  }
  return errSuccess;
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
  // hipFFT does not require buffers at plan creation time.
  (void)in;
  (void)out;

  switch (precision) {
  case shafft::FFTType::C2C:
    plan.fft_type = HIPFFT_C2C;
    break;
  case shafft::FFTType::Z2Z:
    plan.fft_type = HIPFFT_Z2Z;
    break;
  default:
    fprintf(stderr, "Invalid FFT type\n");
    return static_cast<int>(shafft::Status::ERR_INVALID_FFTTYPE);
  }

  for (int i = 0; i < ndim; ++i) {
    if (dimensions[i] <= 0) {
      fprintf(stderr, "Invalid dimension[%d]=%d for hipFFT\n", i, dimensions[i]);
      return errInvalidDim;
    }
  }

  if (diagEnabled()) {
    dbgHdr(plan.comm, "fftndPlan");
    std::cerr << "ndim=" << ndim
              << " precision=" << (plan.fft_type == HIPFFT_Z2Z ? "Z2Z(double)" : "C2C(float)")
              << " ";
    printArray("dimensions", dimensions, ndim);
    std::cerr << " nta=" << nta << " ";
    printArray("ta", ta, nta);
    std::cerr << "\n";
  }

  // Decompose into <=3D chunks
  std::vector<int> subseq(std::max(nta, 1));
  int nsubplansInit = 0;
  transformSubsequence(nta, &nsubplansInit, subseq.data());
  subseq.resize(nsubplansInit);

  plan.nsubplans = static_cast<int>(subseq.size());

  if (diagEnabled()) {
    dbgHdr(plan.comm, "fftndPlan");
    std::cerr << "subsequence=";
    printArray("", subseq.data(), plan.nsubplans);
    std::cerr << "\n";
  }

  // Allocate per-subplan state
  plan.subplans = new hipfftHandle[plan.nsubplans]();
  plan.superbatches = new long long[plan.nsubplans];
  plan.superbatches_offset = new long long[plan.nsubplans];
  plan.innerbatches = new long long[plan.nsubplans];
  plan.innerbatches_offset = new long long[plan.nsubplans];
  plan.transpose_info = new TransposeFFTInfo[plan.nsubplans]();
  plan.methods = new FFTMethod[plan.nsubplans];

  int taTaken = 0;

  // Create each subplan
  for (int j = 0; j < plan.nsubplans; ++j) {
    int r = subseq[j];
    assert(r >= 1 && r <= 3);
    taTaken += r;

    // Get per-rank threshold for this subplan's FFT dimensionality
    const long long sbThreshold = getSuperbatchThreshold(r);

    const int* taWin = &ta[nta - taTaken];

    std::vector<long long> nJ(r);
    long long idist = 0, odist = 0, istride = 0, ostride = 0, batch = 0;
    long long superbatch = 0, superbatchOffset = 0;
    long long innerbatch = 0, innerbatchOffset = 0;

    int tpRc = transformParameters64(ndim,
                                     dimensions,
                                     r,
                                     taWin,
                                     nJ.data(),
                                     &idist,
                                     &odist,
                                     &istride,
                                     &ostride,
                                     &batch,
                                     &superbatch,
                                     &superbatchOffset,
                                     &innerbatch,
                                     &innerbatchOffset);
    if (tpRc != errSuccess)
      return tpRc;

    if (diagEnabled()) {
      dbgHdr(plan.comm, "fftndPlan");
      std::cerr << "subplan j=" << j << " rank=" << r << " taWin=";
      printArray("", taWin, r);
      std::cerr << " n=";
      printArray("", nJ.data(), r);
      std::cerr << " istride=" << istride << " idist=" << idist << " batch=" << batch
                << " superbatch=" << superbatch << " superbatchOffset=" << superbatchOffset
                << " innerbatch=" << innerbatch << " innerbatchOffset=" << innerbatchOffset << "\n";
    }

    plan.superbatches[j] = superbatch;
    plan.superbatches_offset[j] = superbatchOffset;
    plan.innerbatches[j] = innerbatch;
    plan.innerbatches_offset[j] = innerbatchOffset;

    // Check if we should use transpose-based FFT for this subplan
    if (superbatch > sbThreshold) {
      if (diagEnabled()) {
        dbgHdr(plan.comm, "fftndPlan");
        std::cerr << "  superbatch=" << superbatch << " exceeds threshold=" << sbThreshold
                  << ", creating transpose-based FFT\n";
      }

      int rc = createTransposeFFT(ndim,
                                  dimensions,
                                  r,
                                  taWin,
                                  plan.fft_type,
                                  plan.stream,
                                  plan.comm,
                                  plan.transpose_info[j]);
      if (rc != errSuccess)
        return rc;

      if (plan.transpose_info[j].enabled) {
        // Transpose-based FFT successfully created, skip regular plan creation
        plan.subplans[j] = nullptr;
        plan.methods[j] = FFTMethod::TRANSPOSE;
        if (diagEnabled()) {
          dbgHdr(plan.comm, "fftndPlan");
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
    std::vector<long long> inembed(nJ);
    std::vector<long long> onembed(nJ);

    // Use hipfftMakePlanMany64 for 64-bit parameter support
    if (int rc = hipfftCheck(hipfftMakePlanMany64(plan.subplans[j],
                                                  r,
                                                  nJ.data(),
                                                  inembed.data(),
                                                  istride,
                                                  idist,
                                                  onembed.data(),
                                                  ostride,
                                                  odist,
                                                  plan.fft_type,
                                                  batch,
                                                  &worksize)))
      return rc;

    // Record the method used for this subplan
    // TRAILING = FFT on trailing axes (contiguous, maxAxis == ndim-1)
    // STRIDED = FFT on non-trailing axes (requires superbatch loop)
    int maxAxis = *std::max_element(taWin, taWin + r);
    plan.methods[j] = (maxAxis == ndim - 1) ? FFTMethod::TRAILING : FFTMethod::STRIDED;

    if (diagEnabled()) {
      dbgHdr(plan.comm, "fftndPlan");
      std::cerr << "  created plan: worksize=" << worksize
                << " method=" << (plan.methods[j] == FFTMethod::TRAILING ? "TRAILING" : "STRIDED")
                << "\n";
    }
  }
  return 0;
}

int fftndSetStream(FFTNDHandle& plan, hipStream_t stream) {
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

int fftndExecute(FFTNDHandle plan, void*& idata, void*& odata, shafft::FFTDirection direction) {
  int dir = (direction == shafft::FFTDirection::FORWARD) ? HIPFFT_FORWARD : HIPFFT_BACKWARD;

  // Buffer management: track which buffer currently holds the data.
  // - buf[0] = idata (first buffer passed in)
  // - buf[1] = odata (second buffer passed in)
  // - dataInBuf0: true if current data is in buf[0], false if in buf[1]
  //
  // Both regular and transpose paths flip the buffer each subplan:
  // - Regular path: reads from src, writes to dst -> flips
  // - Transpose path: src->dst (transpose) -> dst->src (FFT) -> src->dst (transpose back) -> flips
  //
  // Result location is predictable: buf[1] if nsubplans odd, buf[0] if even.
  void* buf[2] = {idata, odata};
  bool dataInBuf0 = true; // Input starts in buf[0] (idata)

  if (diagEnabled()) {
    dbgHdr(plan.comm, "fftndExecute");
    std::cerr << "begin " << (direction == shafft::FFTDirection::FORWARD ? "FORWARD" : "BACKWARD")
              << " nsubplans=" << plan.nsubplans << " buf[0]=" << buf[0] << " buf[1]=" << buf[1]
              << "\n";
  }

  for (int j = 0; j < plan.nsubplans; ++j) {
    // Determine source (where data currently is) and destination (other buffer)
    void* src = dataInBuf0 ? buf[0] : buf[1];
    void* dst = dataInBuf0 ? buf[1] : buf[0];

    // Check for transpose-based FFT execution
    if (plan.transpose_info && plan.transpose_info[j].enabled) {
      TransposeFFTInfo& ti = plan.transpose_info[j];

      if (diagEnabled()) {
        dbgHdr(plan.comm, "fftndExecute");
        std::cerr << "subplan j=" << j
                  << " (transpose) data_in=" << (dataInBuf0 ? "buf[0]" : "buf[1]") << "\n";
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
        return errRuntime;
      }

      // Step 2: FFT out-of-place dst -> src
      switch (plan.fft_type) {
      case HIPFFT_C2C:
        if (int rc = hipfftCheck(hipfftExecC2C(ti.transposed_fft,
                                               reinterpret_cast<hipfftComplex*>(dst),
                                               reinterpret_cast<hipfftComplex*>(src),
                                               dir)))
          return rc;
        break;
      case HIPFFT_Z2Z:
        if (int rc = hipfftCheck(hipfftExecZ2Z(ti.transposed_fft,
                                               reinterpret_cast<hipfftDoubleComplex*>(dst),
                                               reinterpret_cast<hipfftDoubleComplex*>(src),
                                               dir)))
          return rc;
        break;
      default:
        assert(false && "Unsupported fft_type");
      }

      // Step 3: Transpose back src -> dst (data ends up in dst with original layout)
      gpuRes = gputtExecute(ti.transpose_to_back, src, dst);
      if (gpuRes != GPUTT_SUCCESS) {
        fprintf(stderr, "gputtExecute (to_back) failed for subplan %d with error %d\n", j, gpuRes);
        return errRuntime;
      }

      // Data is now in dst - buffer flip occurred, same as regular path
      dataInBuf0 = !dataInBuf0;

      if (diagEnabled()) {
        dbgHdr(plan.comm, "fftndExecute");
        std::cerr << "  transpose complete, data now in " << (dataInBuf0 ? "buf[0]" : "buf[1]")
                  << "\n";
      }
      continue;
    }

    // Regular (non-transpose) execution path: src -> dst
    const long long sbCount = plan.superbatches[j];
    const long long sbOffset = plan.superbatches_offset[j];
    const long long ibCount = plan.innerbatches[j];
    const long long ibOffset = plan.innerbatches_offset[j];

    if (sbCount <= 0 || sbOffset <= 0) {
      fprintf(stderr, "Invalid superbatch parameters at execution time.\n");
      return errRuntime;
    }

    if (diagEnabled()) {
      dbgHdr(plan.comm, "fftndExecute");
      std::cerr << "subplan j=" << j << " (regular)" << " sb=" << sbCount << "x" << sbOffset
                << " ib=" << ibCount << "x" << ibOffset << " src=" << src << " dst=" << dst << "\n";
    }

    // Execute batched FFTs with superbatch iteration
    for (long long k = 0; k < sbCount; ++k) {
      const size_t sbOff = static_cast<size_t>(k) * static_cast<size_t>(sbOffset);

      for (long long m = 0; m < ibCount; ++m) {
        const size_t off = sbOff + static_cast<size_t>(m) * static_cast<size_t>(ibOffset);

        switch (plan.fft_type) {
        case HIPFFT_C2C:
          if (int rc = hipfftCheck(hipfftExecC2C(plan.subplans[j],
                                                 &reinterpret_cast<hipfftComplex*>(src)[off],
                                                 &reinterpret_cast<hipfftComplex*>(dst)[off],
                                                 dir)))
            return rc;
          break;
        case HIPFFT_Z2Z:
          if (int rc = hipfftCheck(hipfftExecZ2Z(plan.subplans[j],
                                                 &reinterpret_cast<hipfftDoubleComplex*>(src)[off],
                                                 &reinterpret_cast<hipfftDoubleComplex*>(dst)[off],
                                                 dir)))
            return rc;
          break;
        default:
          assert(false && "Unsupported fft_type");
        }
      }
    }

    // Data is now in dst
    dataInBuf0 = !dataInBuf0;

    if (diagEnabled()) {
      dbgHdr(plan.comm, "fftndExecute");
      std::cerr << "  complete, data now in " << (dataInBuf0 ? "buf[0]" : "buf[1]") << "\n";
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

  if (diagEnabled()) {
    dbgHdr(plan.comm, "fftndExecute");
    std::cerr << "end, result in idata=" << idata << "\n";
  }
  return 0;
}

int fftndDestroy(FFTNDHandle plan) {
  int firstError = 0;

  // Destroy transpose-based FFT resources
  if (plan.transpose_info) {
    for (int j = 0; j < plan.nsubplans; ++j) {
      destroyTransposeFFT(plan.transpose_info[j]);
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
      if (res != HIPFFT_SUCCESS && firstError == 0) {
        firstError = static_cast<int>(res);
        fprintf(stderr,
                "hipfftDestroy failed for subplan %d: %d at %s:%d\n",
                j,
                firstError,
                __FILE__,
                __LINE__);
      }
    }
    delete[] plan.subplans;
  }
  delete[] plan.superbatches;
  delete[] plan.superbatches_offset;
  delete[] plan.innerbatches;
  delete[] plan.innerbatches_offset;
  delete[] plan.methods;
  plan.subplans = nullptr;
  plan.superbatches = nullptr;
  plan.superbatches_offset = nullptr;
  plan.innerbatches = nullptr;
  plan.innerbatches_offset = nullptr;
  plan.methods = nullptr;
  plan.nsubplans = 0;
  return firstError;
}
