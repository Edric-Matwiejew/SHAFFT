// transpose_fft.cpp
//
// Transpose-based FFT strategy using gpuTT.
// See transpose_fft.h for interface documentation.
//
#include "transpose_fft.h"

#include <shafft/shafft_error.hpp>

#include "hipcheck.h"

#include <algorithm>
#include <cstdlib>
#include <gputt.h>
#include <iostream>
#include <mpi.h>
#include <vector>

static constexpr int ERR_SUCCESS = static_cast<int>(shafft::Status::SHAFFT_SUCCESS);
static constexpr int ERR_RUNTIME = static_cast<int>(shafft::Status::SHAFFT_ERR_BACKEND);

// ---------------- Diagnostics ----------------
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
  std::cerr << "[SHAFFT:transpose]"
            << (r >= 0 ? ("[r" + std::to_string(r) + "]") : std::string("[r?]")) << "[" << where
            << "] ";
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
  return ERR_RUNTIME;
}

// ---------------- Product helper ----------------
static inline long long prod_range_ll(const int* a, int i0, int i1) {
  long long p = 1;
  if (i0 > i1)
    return 1;
  for (int i = i0; i <= i1; ++i)
    p *= static_cast<long long>(a[i]);
  return p;
}

// ---------------- Public API ----------------

long long get_superbatch_threshold() {
  static long long cached = -1;
  if (cached < 0) {
    const char* env = std::getenv("SHAFFT_SUPERBATCH_THRESHOLD");
    if (env && env[0]) {
      cached = std::atoll(env);
      if (cached < 1)
        cached = 1;
    } else {
      cached = 16;  // Default threshold
    }
  }
  return cached;
}

int create_transpose_fft(int ndim, const int* dimensions, int nta, const int* ta,
                         hipfftType_t fft_type, hipStream_t stream, TransposeFFTInfo& info) {
  info.enabled = false;
  info.rank = ndim;

  // gpuTT requires all dimensions > 1. Check if any dimension is 1.
  // If so, fall back to superbatch iteration.
  for (int i = 0; i < ndim; ++i) {
    if (dimensions[i] <= 1) {
      if (diag_enabled()) {
        dbg_hdr("create_transpose_fft");
        std::cerr << "dimension[" << i << "]=" << dimensions[i]
                  << " <= 1, skipping transpose path (gpuTT limitation)\n";
      }
      return ERR_SUCCESS;
    }
  }

  // Find contiguous span of FFT axes
  std::vector<int> axes_sorted(ta, ta + nta);
  std::sort(axes_sorted.begin(), axes_sorted.end());
  const int minAxis = axes_sorted.front();
  const int maxAxis = axes_sorted.back();

  if (maxAxis == ndim - 1) {
    // Already trailing - no transpose needed
    return ERR_SUCCESS;
  }

  // Build permutation to bring FFT axes to the end
  // Original layout: [0, 1, ..., minAxis-1, minAxis, ..., maxAxis, maxAxis+1, ..., ndim-1]
  // Transposed:      [0, 1, ..., minAxis-1, maxAxis+1, ..., ndim-1, minAxis, ..., maxAxis]
  //
  // This makes the FFT axes contiguous and trailing.
  std::vector<int> perm_to_front(ndim);
  std::vector<int> perm_to_back(ndim);
  std::vector<int> transposed_dims(ndim);

  int write_idx = 0;
  // First: leading dims [0..minAxis-1] stay in place
  for (int i = 0; i < minAxis; ++i) {
    perm_to_front[write_idx++] = i;
  }
  // Second: trailing dims [maxAxis+1..ndim-1] move forward
  for (int i = maxAxis + 1; i < ndim; ++i) {
    perm_to_front[write_idx++] = i;
  }
  // Third: FFT dims [minAxis..maxAxis] become trailing
  for (int i = minAxis; i <= maxAxis; ++i) {
    perm_to_front[write_idx++] = i;
  }

  // Compute inverse permutation for transpose back
  for (int i = 0; i < ndim; ++i) {
    perm_to_back[perm_to_front[i]] = i;
  }

  // Compute transposed dimensions
  for (int i = 0; i < ndim; ++i) {
    transposed_dims[i] = dimensions[perm_to_front[i]];
  }

  if (diag_enabled()) {
    dbg_hdr("create_transpose_fft");
    std::cerr << "minAxis=" << minAxis << " maxAxis=" << maxAxis << " ";
    print_array("perm_to_front", perm_to_front.data(), ndim);
    std::cerr << " ";
    print_array("perm_to_back", perm_to_back.data(), ndim);
    std::cerr << " ";
    print_array("transposed_dims", transposed_dims.data(), ndim);
    std::cerr << "\n";
  }

  // Store transposed dimensions for later use
  info.transposed_dims = new int[ndim];
  for (int i = 0; i < ndim; ++i) {
    info.transposed_dims[i] = transposed_dims[i];
  }

  // Determine gpuTT data type based on FFT type
  gputtDataType dtype = (fft_type == HIPFFT_Z2Z)
                            ? gputtDataTypeFloat64   // Z2Z uses double complex = 2x double
                            : gputtDataTypeFloat32;  // C2C uses float complex = 2x float
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

  int complex_rank = ndim + 1;
  std::vector<int> complex_dims(complex_rank);
  std::vector<int> complex_perm_fwd(complex_rank);
  std::vector<int> complex_perm_bwd(complex_rank);

  // Build column-major dims: [realimag=2, dim(n-1), dim(n-2), ..., dim0]
  complex_dims[0] = 2;  // Real/imag is fastest
  for (int i = 0; i < ndim; ++i) {
    complex_dims[i + 1] = dimensions[ndim - 1 - i];  // Reverse order
  }

  // Build column-major permutations.
  // Row-major perm_to_front: row_i -> row_perm[i]
  // We need column-major perm where col_j -> col_perm[j]
  // col_j corresponds to row_(ndim-1-j) for j in [1..ndim]
  // col 0 is realimag, always stays at 0
  complex_perm_fwd[0] = 0;  // realimag stays fastest
  complex_perm_bwd[0] = 0;

  for (int col = 1; col <= ndim; ++col) {
    // col corresponds to row dimension (ndim - col)
    int row_idx = ndim - col;
    int row_perm_val = perm_to_front[row_idx];
    // row_perm_val becomes column index (ndim - row_perm_val)
    int col_perm_val = ndim - row_perm_val;
    complex_perm_fwd[col] = col_perm_val;
  }

  // Inverse permutation
  for (int col = 0; col <= ndim; ++col) {
    complex_perm_bwd[complex_perm_fwd[col]] = col;
  }

  // Column-major transposed dims: output_dims[i] = input_dims[perm[i]]
  std::vector<int> complex_transposed_dims(complex_rank);
  for (int col = 0; col <= ndim; ++col) {
    complex_transposed_dims[col] = complex_dims[complex_perm_fwd[col]];
  }

  // ========== SQUEEZE OUT SIZE-1 DIMENSIONS ==========
  // gpuTT requires all dimensions > 1. Dimensions of size 1 don't affect
  // memory layout, so we can safely remove them and adjust the permutation.
  //
  // Strategy:
  // 1. Identify which indices have size > 1
  // 2. Build mapping from old indices to new (squeezed) indices
  // 3. Create squeezed dim arrays and adjusted permutations

  // For forward transpose: squeeze complex_dims
  std::vector<int> squeeze_map_fwd(complex_rank, -1);  // old_idx -> new_idx, -1 if squeezed
  std::vector<int> squeezed_dims_fwd;
  std::vector<int> squeezed_perm_fwd;

  for (int i = 0; i < complex_rank; ++i) {
    if (complex_dims[i] > 1) {
      squeeze_map_fwd[i] = static_cast<int>(squeezed_dims_fwd.size());
      squeezed_dims_fwd.push_back(complex_dims[i]);
    }
  }

  // Adjust forward permutation: only include non-squeezed dimensions
  for (int i = 0; i < complex_rank; ++i) {
    if (squeeze_map_fwd[i] >= 0) {  // This dimension is kept
      int target = complex_perm_fwd[i];
      // The target must also be kept (size-1 dims map to size-1 dims in a valid perm)
      if (squeeze_map_fwd[target] >= 0) {
        squeezed_perm_fwd.push_back(squeeze_map_fwd[target]);
      }
    }
  }

  // For backward transpose: squeeze complex_transposed_dims (input of backward)
  // The backward transpose has:
  //   - Input: transposed layout (complex_transposed_dims)
  //   - Output: original layout (complex_dims)
  // So squeeze_map_bwd maps indices in the transposed layout,
  // and we need squeeze_map_fwd to map the output (original layout) indices.
  std::vector<int> squeeze_map_bwd(complex_rank, -1);
  std::vector<int> squeezed_dims_bwd;
  std::vector<int> squeezed_perm_bwd;

  for (int i = 0; i < complex_rank; ++i) {
    if (complex_transposed_dims[i] > 1) {
      squeeze_map_bwd[i] = static_cast<int>(squeezed_dims_bwd.size());
      squeezed_dims_bwd.push_back(complex_transposed_dims[i]);
    }
  }

  // Adjust backward permutation
  // gpuTT semantics: output[i] = input[perm[i]]
  // complex_perm_bwd[i] gives the input index in transposed layout
  // that should go to output index i in original layout.
  // We iterate over output indices (original layout) and build the squeezed perm.
  for (int i = 0; i < complex_rank; ++i) {
    if (squeeze_map_fwd[i] >= 0) {         // Output index i is kept (original layout)
      int source = complex_perm_bwd[i];    // Source is in transposed layout
      if (squeeze_map_bwd[source] >= 0) {  // Source index is also kept
        squeezed_perm_bwd.push_back(squeeze_map_bwd[source]);
      }
    }
  }

  int squeezed_rank_fwd = static_cast<int>(squeezed_dims_fwd.size());
  int squeezed_rank_bwd = static_cast<int>(squeezed_dims_bwd.size());

  if (diag_enabled()) {
    dbg_hdr("create_transpose_fft");
    std::cerr << "gpuTT column-major: ";
    print_array("dims", complex_dims.data(), complex_rank);
    std::cerr << " ";
    print_array("perm_fwd", complex_perm_fwd.data(), complex_rank);
    std::cerr << " ";
    print_array("perm_bwd", complex_perm_bwd.data(), complex_rank);
    std::cerr << " ";
    print_array("transposed_dims", complex_transposed_dims.data(), complex_rank);
    std::cerr << "\n";

    dbg_hdr("create_transpose_fft");
    std::cerr << "squeezed fwd: ";
    print_array("dims", squeezed_dims_fwd.data(), squeezed_rank_fwd);
    std::cerr << " ";
    print_array("perm", squeezed_perm_fwd.data(), squeezed_rank_fwd);
    std::cerr << " squeezed bwd: ";
    print_array("dims", squeezed_dims_bwd.data(), squeezed_rank_bwd);
    std::cerr << " ";
    print_array("perm", squeezed_perm_bwd.data(), squeezed_rank_bwd);
    std::cerr << "\n";
  }

  // Sanity check: squeezed permutations should have same size as squeezed dims
  if (static_cast<int>(squeezed_perm_fwd.size()) != squeezed_rank_fwd ||
      static_cast<int>(squeezed_perm_bwd.size()) != squeezed_rank_bwd) {
    if (diag_enabled()) {
      dbg_hdr("create_transpose_fft");
      std::cerr << "ERROR: squeezed permutation size mismatch!\n";
    }
    delete[] info.transposed_dims;
    info.transposed_dims = nullptr;
    return ERR_RUNTIME;
  }

  // If after squeezing we have rank < 2, the transpose is trivial (identity)
  if (squeezed_rank_fwd < 2) {
    if (diag_enabled()) {
      dbg_hdr("create_transpose_fft");
      std::cerr << "squeezed rank < 2, transpose is identity, skipping\n";
    }
    delete[] info.transposed_dims;
    info.transposed_dims = nullptr;
    return ERR_SUCCESS;  // info.enabled remains false
  }

  // Create transpose plan: original layout -> FFT axes trailing
  gputtResult res = gputtPlan(&info.transpose_to_front, squeezed_rank_fwd, squeezed_dims_fwd.data(),
                              squeezed_perm_fwd.data(), dtype, static_cast<gputtStream>(stream));
  if (res != GPUTT_SUCCESS) {
    delete[] info.transposed_dims;
    info.transposed_dims = nullptr;
    return gputtCheck(res);
  }

  // Create inverse transpose plan: FFT axes trailing -> original layout
  // Use squeezed dimensions for backward transpose as well
  res = gputtPlan(&info.transpose_to_back, squeezed_rank_bwd, squeezed_dims_bwd.data(),
                  squeezed_perm_bwd.data(), dtype, static_cast<gputtStream>(stream));
  if (res != GPUTT_SUCCESS) {
    gputtDestroy(info.transpose_to_front);
    info.transpose_to_front = nullptr;
    delete[] info.transposed_dims;
    info.transposed_dims = nullptr;
    return gputtCheck(res);
  }

  // Create FFT plan for transposed layout (FFT axes are now trailing)
  // In transposed layout: FFT axes are at [ndim-nta..ndim-1]
  std::vector<long long> n_fft(nta);
  for (int i = 0; i < nta; ++i) {
    n_fft[i] = static_cast<long long>(transposed_dims[ndim - nta + i]);
  }

  // Compute FFT parameters for trailing-axis FFT (simple case)
  long long fft_size = prod_range_ll(transposed_dims.data(), ndim - nta, ndim - 1);
  long long total_size = prod_range_ll(transposed_dims.data(), 0, ndim - 1);
  long long batch = total_size / fft_size;

  if (int rc = hipfftCheck(hipfftCreate(&info.transposed_fft))) {
    gputtDestroy(info.transpose_to_front);
    gputtDestroy(info.transpose_to_back);
    info.transpose_to_front = nullptr;
    info.transpose_to_back = nullptr;
    delete[] info.transposed_dims;
    info.transposed_dims = nullptr;
    return rc;
  }

  if (int rc = hipfftCheck(hipfftSetStream(info.transposed_fft, stream))) {
    hipfftDestroy(info.transposed_fft);
    gputtDestroy(info.transpose_to_front);
    gputtDestroy(info.transpose_to_back);
    info.transposed_fft = nullptr;
    info.transpose_to_front = nullptr;
    info.transpose_to_back = nullptr;
    delete[] info.transposed_dims;
    info.transposed_dims = nullptr;
    return rc;
  }

  size_t worksize = 0;
  std::vector<long long> inembed(n_fft);
  std::vector<long long> onembed(n_fft);

  // For trailing axes: istride=1, idist=fft_size
  if (int rc = hipfftCheck(hipfftMakePlanMany64(info.transposed_fft, nta, n_fft.data(),
                                                inembed.data(), 1LL, fft_size, onembed.data(), 1LL,
                                                fft_size, fft_type, batch, &worksize))) {
    hipfftDestroy(info.transposed_fft);
    gputtDestroy(info.transpose_to_front);
    gputtDestroy(info.transpose_to_back);
    info.transposed_fft = nullptr;
    info.transpose_to_front = nullptr;
    info.transpose_to_back = nullptr;
    delete[] info.transposed_dims;
    info.transposed_dims = nullptr;
    return rc;
  }

  if (diag_enabled()) {
    dbg_hdr("create_transpose_fft");
    std::cerr << "transposed FFT: nta=" << nta;
    print_array(" n", n_fft.data(), nta);
    std::cerr << " batch=" << batch << " fft_size=" << fft_size << " worksize=" << worksize << "\n";
  }

  info.enabled = true;
  return ERR_SUCCESS;
}

void destroy_transpose_fft(TransposeFFTInfo& info) {
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
  if (info.transposed_dims) {
    delete[] info.transposed_dims;
    info.transposed_dims = nullptr;
  }
  info.enabled = false;
  info.rank = 0;
}
