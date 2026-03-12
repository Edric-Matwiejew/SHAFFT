// Transpose-based FFT strategy using gpuTT.
#include "transpose_fftnd.hpp"

#include <shafft/shafft_error.hpp>

#include "../../../detail/array_utils.hpp"
#include "../../common/diag.hpp"
#include "../common/hipcheck.hpp"

#include <algorithm>
#include <cstdlib>
#include <gputt.h>
#include <iostream>
#include <mpi.h>
#include <vector>

static constexpr int errSuccess = static_cast<int>(shafft::Status::SUCCESS);
static constexpr int errRuntime = static_cast<int>(shafft::Status::ERR_BACKEND);

using shafft::fft::diagEnabled;
using shafft::fft::mpiRank;
using shafft::fft::printArray;

// Module-specific debug header
static void dbgHdr(MPI_Comm comm, const char* where) {
  shafft::fft::dbgHdr("ND-gpuTT", comm, where);
}

// ---------------- gpuTT error checking ----------------
static int gputtCheck(gputtResult res, const char* file = __FILE__, int line = __LINE__) {
  if (res == GPUTT_SUCCESS)
    return 0;
  const char* msg = "Unknown gpuTT error";
  switch (res) {
  case GPUTT_INVALID_PLAN:
    msg = "Invalid gpuTT plan";
    break;
  case GPUTT_INVALID_PARAMETER:
    msg = "Invalid gpuTT parameter";
    break;
  case GPUTT_INVALID_DEVICE:
    msg = "Invalid gpuTT device";
    break;
  case GPUTT_INTERNAL_ERROR:
    msg = "gpuTT internal error";
    break;
  case GPUTT_UNDEFINED_ERROR:
    msg = "gpuTT undefined error";
    break;
  case GPUTT_UNSUPPORTED_METHOD:
    msg = "gpuTT unsupported method";
    break;
  default:
    break;
  }
  fprintf(stderr, "gpuTT error: %s at %s:%d\n", msg, file, line);
  return errRuntime;
}

// ---------------- Superbatch threshold cache (file-scope for resetability) ----------------
namespace {
long long g_cached1d = -1;
long long g_cached2d = -1;
long long g_cached3d = -1;
long long g_cachedLegacy = -1;
bool g_initialized = false;
} // namespace

// ---------------- Public API ----------------

long long getSuperbatchThreshold(int fftRank) {
  // Per-rank thresholds based on MI250X benchmarks (2026-02-03):
  //   1D: threshold=16 gives 64.7% improvement over always-strided
  //   2D: threshold=16 gives 29.0% improvement over always-strided
  //   3D: threshold=1 (always transpose) gives 84.9% improvement

  if (!g_initialized) {
    // Check legacy environment variable first (overrides all)
    const char* envLegacy = std::getenv("SHAFFT_SUPERBATCH_THRESHOLD");
    if (envLegacy && envLegacy[0]) {
      g_cachedLegacy = std::atoll(envLegacy);
      if (g_cachedLegacy < 1)
        g_cachedLegacy = 1;
    }

    // Per-rank environment variables
    const char* env1d = std::getenv("SHAFFT_SUPERBATCH_THRESHOLD_1D");
    const char* env2d = std::getenv("SHAFFT_SUPERBATCH_THRESHOLD_2D");
    const char* env3d = std::getenv("SHAFFT_SUPERBATCH_THRESHOLD_3D");

    // 1D threshold (default: 16)
    if (env1d && env1d[0]) {
      g_cached1d = std::atoll(env1d);
      if (g_cached1d < 1)
        g_cached1d = 1;
    } else {
      g_cached1d = 16;
    }

    // 2D threshold (default: 16)
    if (env2d && env2d[0]) {
      g_cached2d = std::atoll(env2d);
      if (g_cached2d < 1)
        g_cached2d = 1;
    } else {
      g_cached2d = 16;
    }

    // 3D threshold (default: 1, always use transpose)
    if (env3d && env3d[0]) {
      g_cached3d = std::atoll(env3d);
      if (g_cached3d < 1)
        g_cached3d = 1;
    } else {
      g_cached3d = 1;
    }

    g_initialized = true;
  }

  // Legacy override takes precedence
  if (g_cachedLegacy > 0) {
    return g_cachedLegacy;
  }

  // Return per-rank threshold
  switch (fftRank) {
  case 1:
    return g_cached1d;
  case 2:
    return g_cached2d;
  case 3:
    return g_cached3d;
  default:
    return g_cached2d; // Fallback for unexpected ranks
  }
}

void resetSuperbatchThresholdCache() {
  // Reset all cached values - next call to getSuperbatchThreshold() will re-read env vars
  g_cached1d = -1;
  g_cached2d = -1;
  g_cached3d = -1;
  g_cachedLegacy = -1;
  g_initialized = false;
}

int createTransposeFFT(int ndim,
                       const int* dimensions,
                       int nta,
                       const int* ta,
                       hipfftType_t fftType,
                       hipStream_t stream,
                       MPI_Comm comm,
                       TransposeFFTInfo& info) {
  info.enabled = false;
  info.rank = ndim;

  // gpuTT requires all dimensions > 1. Check if any dimension is 1.
  // If so, fall back to superbatch iteration.
  for (int i = 0; i < ndim; ++i) {
    if (dimensions[i] <= 1) {
      if (diagEnabled()) {
        dbgHdr(comm, "createTransposeFFT");
        std::cerr << "dimension[" << i << "]=" << dimensions[i]
                  << " <= 1, skipping transpose path (gpuTT limitation)\n";
      }
      return errSuccess;
    }
  }

  // Find contiguous span of FFT axes
  std::vector<int> axesSorted(ta, ta + nta);
  std::sort(axesSorted.begin(), axesSorted.end());
  const int minAxis = axesSorted.front();
  const int maxAxis = axesSorted.back();

  if (maxAxis == ndim - 1) {
    // Already trailing - no transpose needed
    return errSuccess;
  }

  // Build permutation to bring FFT axes to the end
  // Original layout: [0, 1, ..., minAxis-1, minAxis, ..., maxAxis, maxAxis+1, ..., ndim-1]
  // Transposed:      [0, 1, ..., minAxis-1, maxAxis+1, ..., ndim-1, minAxis, ..., maxAxis]
  //
  // This makes the FFT axes contiguous and trailing.
  std::vector<int> permToFront(ndim);
  std::vector<int> permToBack(ndim);
  std::vector<int> transposedDims(ndim);

  int writeIdx = 0;
  // First: leading dims [0..minAxis-1] stay in place
  for (int i = 0; i < minAxis; ++i) {
    permToFront[writeIdx++] = i;
  }
  // Second: trailing dims [maxAxis+1..ndim-1] move forward
  for (int i = maxAxis + 1; i < ndim; ++i) {
    permToFront[writeIdx++] = i;
  }
  // Third: FFT dims [minAxis..maxAxis] become trailing
  for (int i = minAxis; i <= maxAxis; ++i) {
    permToFront[writeIdx++] = i;
  }

  // Compute inverse permutation for transpose back
  for (int i = 0; i < ndim; ++i) {
    permToBack[permToFront[i]] = i;
  }

  // Compute transposed dimensions
  for (int i = 0; i < ndim; ++i) {
    transposedDims[i] = static_cast<int>(dimensions[permToFront[i]]);
  }

  if (diagEnabled()) {
    dbgHdr(comm, "createTransposeFFT");
    std::cerr << "minAxis=" << minAxis << " maxAxis=" << maxAxis << " ";
    printArray("permToFront", permToFront.data(), ndim);
    std::cerr << " ";
    printArray("permToBack", permToBack.data(), ndim);
    std::cerr << " ";
    printArray("transposedDims", transposedDims.data(), ndim);
    std::cerr << "\n";
  }

  // Store transposed dimensions for later use
  info.transposedDims = new int[ndim];
  for (int i = 0; i < ndim; ++i) {
    info.transposedDims[i] = transposedDims[i];
  }

  // Determine gpuTT data type based on FFT type
  gputtDataType dtype = (fftType == HIPFFT_Z2Z)
                            ? gputtDataTypeFloat64  // Z2Z uses double complex = 2x double
                            : gputtDataTypeFloat32; // C2C uses float complex = 2x float
  // Note: gpuTT treats complex as 2 consecutive real elements, so we adjust dimensions

  // CRITICAL: gpuTT uses COLUMN-MAJOR ordering where dims[0] is the FASTEST varying.
  // Our data is C/C++ row-major where the LAST dimension is fastest.
  // For complex data with real/imag interleaved, the complex dimension (size 2)
  // is the very fastest (innermost).
  //
  // To convert our row-major layout to gpuTT's column-major convention:
  //   Row-major dims: [dim0, dim1, ..., dim(n-1)] where dim(n-1) is fastest
  //   gpuTT column-major: [realimag=2, dim(n-1), dim(n-2), ..., dim0]
  //
  // The permutation also needs to be reversed and adjusted.
  //
  // Example: row-major [2, 4, 3] with perm [0, 2, 1] (swap dims 1 and 2)
  //   gpuTT dims: [2, 3, 4, 2] (fastest to slowest)
  //   gpuTT perm to swap j and k: [0, 2, 1, 3] (swap column-major dims 1 and 2)
  //
  // ADDITIONAL: gpuTT requires all dimensions > 1. We "squeeze out" size-1 dims
  // since they don't affect memory layout.

  int complexRank = ndim + 1;
  std::vector<int> complexDims(complexRank);
  std::vector<int> complexPermFwd(complexRank);
  std::vector<int> complexPermBwd(complexRank);

  // Build column-major dims: [realimag=2, dim(n-1), dim(n-2), ..., dim0]
  complexDims[0] = 2; // Real/imag is fastest
  for (int i = 0; i < ndim; ++i) {
    complexDims[i + 1] = static_cast<int>(dimensions[ndim - 1 - i]); // Reverse order
  }

  // Build column-major permutations.
  // Row-major permToFront: row_i -> row_perm[i]
  // We need column-major perm where col_j -> col_perm[j]
  // col_j corresponds to row_(ndim-1-j) for j in [1..ndim]
  // col 0 is realimag, always stays at 0
  complexPermFwd[0] = 0; // realimag stays fastest
  complexPermBwd[0] = 0;

  for (int col = 1; col <= ndim; ++col) {
    // col corresponds to row dimension (ndim - col)
    int rowIdx = ndim - col;
    int rowPermVal = permToFront[rowIdx];
    // rowPermVal becomes column index (ndim - rowPermVal)
    int colPermVal = ndim - rowPermVal;
    complexPermFwd[col] = colPermVal;
  }

  // Inverse permutation
  for (int col = 0; col <= ndim; ++col) {
    complexPermBwd[complexPermFwd[col]] = col;
  }

  // Column-major transposed dims: output_dims[i] = input_dims[perm[i]]
  std::vector<int> complexTransposedDims(complexRank);
  for (int col = 0; col <= ndim; ++col) {
    complexTransposedDims[col] = complexDims[complexPermFwd[col]];
  }

  // ========== SQUEEZE OUT SIZE-1 DIMENSIONS ==========
  // gpuTT requires all dimensions > 1. Dimensions of size 1 don't affect
  // memory layout, so we can safely remove them and adjust the permutation.
  //
  // Strategy:
  // 1. Identify which indices have size > 1
  // 2. Build mapping from old indices to new (squeezed) indices
  // 3. Create squeezed dim arrays and adjusted permutations

  // For forward transpose: squeeze complexDims
  std::vector<int> squeezeMapFwd(complexRank, -1); // old_idx -> new_idx, -1 if squeezed
  std::vector<int> squeezedDimsFwd;
  std::vector<int> squeezedPermFwd;

  for (int i = 0; i < complexRank; ++i) {
    if (complexDims[i] > 1) {
      squeezeMapFwd[i] = static_cast<int>(squeezedDimsFwd.size());
      squeezedDimsFwd.push_back(complexDims[i]);
    }
  }

  // Adjust forward permutation: only include non-squeezed dimensions
  for (int i = 0; i < complexRank; ++i) {
    if (squeezeMapFwd[i] >= 0) { // This dimension is kept
      int target = complexPermFwd[i];
      // The target must also be kept (size-1 dims map to size-1 dims in a valid perm)
      if (squeezeMapFwd[target] >= 0) {
        squeezedPermFwd.push_back(squeezeMapFwd[target]);
      }
    }
  }

  // For backward transpose: squeeze complexTransposedDims (input of backward)
  // The backward transpose has:
  //   - Input: transposed layout (complexTransposedDims)
  //   - Output: original layout (complexDims)
  // So squeezeMapBwd maps indices in the transposed layout,
  // and we need squeezeMapFwd to map the output (original layout) indices.
  std::vector<int> squeezeMapBwd(complexRank, -1);
  std::vector<int> squeezedDimsBwd;
  std::vector<int> squeezedPermBwd;

  for (int i = 0; i < complexRank; ++i) {
    if (complexTransposedDims[i] > 1) {
      squeezeMapBwd[i] = static_cast<int>(squeezedDimsBwd.size());
      squeezedDimsBwd.push_back(complexTransposedDims[i]);
    }
  }

  // Adjust backward permutation
  // gpuTT semantics: output[i] = input[perm[i]]
  // complexPermBwd[i] gives the input index in transposed layout
  // that should go to output index i in original layout.
  // We iterate over output indices (original layout) and build the squeezed perm.
  for (int i = 0; i < complexRank; ++i) {
    if (squeezeMapFwd[i] >= 0) {        // Output index i is kept (original layout)
      int source = complexPermBwd[i];   // Source is in transposed layout
      if (squeezeMapBwd[source] >= 0) { // Source index is also kept
        squeezedPermBwd.push_back(squeezeMapBwd[source]);
      }
    }
  }

  int squeezedRankFwd = static_cast<int>(squeezedDimsFwd.size());
  int squeezedRankBwd = static_cast<int>(squeezedDimsBwd.size());

  if (diagEnabled()) {
    dbgHdr(comm, "createTransposeFFT");
    std::cerr << "gpuTT column-major: ";
    printArray("dims", complexDims.data(), complexRank);
    std::cerr << " ";
    printArray("perm_fwd", complexPermFwd.data(), complexRank);
    std::cerr << " ";
    printArray("perm_bwd", complexPermBwd.data(), complexRank);
    std::cerr << " ";
    printArray("transposedDims", complexTransposedDims.data(), complexRank);
    std::cerr << "\n";

    dbgHdr(comm, "createTransposeFFT");
    std::cerr << "squeezed fwd: ";
    printArray("dims", squeezedDimsFwd.data(), squeezedRankFwd);
    std::cerr << " ";
    printArray("perm", squeezedPermFwd.data(), squeezedRankFwd);
    std::cerr << " squeezed bwd: ";
    printArray("dims", squeezedDimsBwd.data(), squeezedRankBwd);
    std::cerr << " ";
    printArray("perm", squeezedPermBwd.data(), squeezedRankBwd);
    std::cerr << "\n";
  }

  // Sanity check: squeezed permutations should have same size as squeezed dims
  if (static_cast<int>(squeezedPermFwd.size()) != squeezedRankFwd ||
      static_cast<int>(squeezedPermBwd.size()) != squeezedRankBwd) {
    if (diagEnabled()) {
      dbgHdr(comm, "createTransposeFFT");
      std::cerr << "ERROR: squeezed permutation size mismatch!\n";
    }
    delete[] info.transposedDims;
    info.transposedDims = nullptr;
    return errRuntime;
  }

  // If after squeezing we have rank < 2, the transpose is trivial (identity)
  if (squeezedRankFwd < 2) {
    if (diagEnabled()) {
      dbgHdr(comm, "createTransposeFFT");
      std::cerr << "squeezed rank < 2, transpose is identity, skipping\n";
    }
    delete[] info.transposedDims;
    info.transposedDims = nullptr;
    return errSuccess; // info.enabled remains false
  }

  // Create transpose plan: original layout -> FFT axes trailing
  gputtResult res = gputtPlan(&info.transpose_to_front,
                              squeezedRankFwd,
                              squeezedDimsFwd.data(),
                              squeezedPermFwd.data(),
                              dtype,
                              static_cast<gputtStream>(stream));
  if (res != GPUTT_SUCCESS) {
    delete[] info.transposedDims;
    info.transposedDims = nullptr;
    return gputtCheck(res);
  }

  // Create inverse transpose plan: FFT axes trailing -> original layout
  // Use squeezed dimensions for backward transpose as well
  res = gputtPlan(&info.transpose_to_back,
                  squeezedRankBwd,
                  squeezedDimsBwd.data(),
                  squeezedPermBwd.data(),
                  dtype,
                  static_cast<gputtStream>(stream));
  if (res != GPUTT_SUCCESS) {
    gputtDestroy(info.transpose_to_front);
    info.transpose_to_front = nullptr;
    delete[] info.transposedDims;
    info.transposedDims = nullptr;
    return gputtCheck(res);
  }

  // Create FFT plan for transposed layout (FFT axes are now trailing)
  // In transposed layout: FFT axes are at [ndim-nta..ndim-1]
  std::vector<long long> nFFT(nta);
  for (int i = 0; i < nta; ++i) {
    nFFT[i] = static_cast<long long>(transposedDims[ndim - nta + i]);
  }

  // Compute FFT parameters for trailing-axis FFT (simple case)
  long long fftSize =
      shafft::detail::prodRange<int, long long>(transposedDims.data(), ndim - nta, ndim - 1);
  long long totalSize =
      shafft::detail::prodRange<int, long long>(transposedDims.data(), 0, ndim - 1);
  long long batch = totalSize / fftSize;

  if (int rc = hipfftCheck(hipfftCreate(&info.transposed_fft))) {
    gputtDestroy(info.transpose_to_front);
    gputtDestroy(info.transpose_to_back);
    info.transpose_to_front = nullptr;
    info.transpose_to_back = nullptr;
    delete[] info.transposedDims;
    info.transposedDims = nullptr;
    return rc;
  }

  if (int rc = hipfftCheck(hipfftSetStream(info.transposed_fft, stream))) {
    hipfftDestroy(info.transposed_fft);
    gputtDestroy(info.transpose_to_front);
    gputtDestroy(info.transpose_to_back);
    info.transposed_fft = nullptr;
    info.transpose_to_front = nullptr;
    info.transpose_to_back = nullptr;
    delete[] info.transposedDims;
    info.transposedDims = nullptr;
    return rc;
  }

  size_t worksize = 0;
  std::vector<long long> inembed(nFFT);
  std::vector<long long> onembed(nFFT);

  // For trailing axes: istride=1, idist=fftSize
  if (int rc = hipfftCheck(hipfftMakePlanMany64(info.transposed_fft,
                                                nta,
                                                nFFT.data(),
                                                inembed.data(),
                                                1LL,
                                                fftSize,
                                                onembed.data(),
                                                1LL,
                                                fftSize,
                                                fftType,
                                                batch,
                                                &worksize))) {
    hipfftDestroy(info.transposed_fft);
    gputtDestroy(info.transpose_to_front);
    gputtDestroy(info.transpose_to_back);
    info.transposed_fft = nullptr;
    info.transpose_to_front = nullptr;
    info.transpose_to_back = nullptr;
    delete[] info.transposedDims;
    info.transposedDims = nullptr;
    return rc;
  }

  if (diagEnabled()) {
    dbgHdr(comm, "createTransposeFFT");
    std::cerr << "transposed FFT: nta=" << nta;
    printArray(" n", nFFT.data(), nta);
    std::cerr << " batch=" << batch << " fftSize=" << fftSize << " worksize=" << worksize << "\n";
  }

  info.enabled = true;
  return errSuccess;
}

void destroyTransposeFFT(TransposeFFTInfo& info) {
  if (info.transpose_to_front) {
    gputtDestroy(info.transpose_to_front);
    info.transpose_to_front = nullptr;
  }
  if (info.transpose_to_back) {
    gputtDestroy(info.transpose_to_back);
    info.transpose_to_back = nullptr;
  }
  if (info.transposed_fft) {
    hipfftDestroy(info.transposed_fft);
    info.transposed_fft = nullptr;
  }
  if (info.transposedDims) {
    delete[] info.transposedDims;
    info.transposedDims = nullptr;
  }
  info.enabled = false;
  info.rank = 0;
}
