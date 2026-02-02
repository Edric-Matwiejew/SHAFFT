// test_transpose_fft.cpp - Test gpuTT transpose + FFT pattern
//
// Tests the transpose-FFT-transpose pattern used for high-dimensional FFTs
// when the superbatch count exceeds the threshold.

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>
#include <gputt.h>
#include <cstdio>
#include <cmath>
#include <complex>
#include <vector>
#include <mpi.h>

#define HIP_CHECK(x) do { \
  hipError_t err = (x); \
  if (err != hipSuccess) { \
    fprintf(stderr, "HIP error: %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
    return 1; \
  } \
} while(0)

#define HIPFFT_CHECK(x) do { \
  hipfftResult_t err = (x); \
  if (err != HIPFFT_SUCCESS) { \
    fprintf(stderr, "hipFFT error: %d at %s:%d\n", (int)err, __FILE__, __LINE__); \
    return 1; \
  } \
} while(0)

#define GPUTT_CHECK(x) do { \
  gputtResult err = (x); \
  if (err != GPUTT_SUCCESS) { \
    fprintf(stderr, "gpuTT error: %d at %s:%d\n", (int)err, __FILE__, __LINE__); \
    return 1; \
  } \
} while(0)

using Complex = std::complex<double>;

// Simple 4D test case: [2, 3, 4, 5] with FFT on axes [1, 2]
// This has leading dim [0] and trailing dim [3]
// superbatch = 2 (product of leading dims)
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  
  // 4D tensor: [2, 3, 4, 5]
  const int ndim = 4;
  const int dims[4] = {2, 3, 4, 5};
  const int fft_axes[2] = {1, 2};  // FFT on axes 1 and 2 (sizes 3 and 4)
  const int nta = 2;
  
  // Calculate sizes
  long long total_size = 1;
  for (int i = 0; i < ndim; ++i) total_size *= dims[i];
  printf("Testing 4D [%d,%d,%d,%d] with FFT on axes [%d,%d]\n",
         dims[0], dims[1], dims[2], dims[3], fft_axes[0], fft_axes[1]);
  printf("Total elements: %lld\n", total_size);
  
  // Allocate host memory
  std::vector<Complex> h_input(total_size);
  std::vector<Complex> h_output(total_size);
  std::vector<Complex> h_reference(total_size);
  
  // Initialize with test pattern
  for (long long i = 0; i < total_size; ++i) {
    h_input[i] = Complex(cos(i * 0.1), sin(i * 0.1));
  }
  
  // Allocate device memory
  Complex *d_data, *d_work, *d_temp;
  HIP_CHECK(hipMalloc(&d_data, total_size * sizeof(Complex)));
  HIP_CHECK(hipMalloc(&d_work, total_size * sizeof(Complex)));
  HIP_CHECK(hipMalloc(&d_temp, total_size * sizeof(Complex)));
  
  // Copy input to device
  HIP_CHECK(hipMemcpy(d_data, h_input.data(), total_size * sizeof(Complex), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_work, h_input.data(), total_size * sizeof(Complex), hipMemcpyHostToDevice));
  
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  
  // ===== Reference: direct strided FFT with superbatch iteration =====
  {
    // FFT on axes [1,2] with dims [3,4]
    // istride = product of trailing dims = 5
    // idist = product of FFT dims * trailing = 3*4*5 = 60
    // batch = trailing product = 5
    // superbatch = leading product = 2
    
    long long n[2] = {3, 4};  // slowest to fastest in FFT block
    long long istride = 5;    // trailing product
    long long idist = 1;      // distance between batch elements (for non-trailing)
    long long batch = 5;      // trailing product
    long long superbatch_offset = 60;  // product of FFT + trailing dims
    
    hipfftHandle plan;
    HIPFFT_CHECK(hipfftCreate(&plan));
    HIPFFT_CHECK(hipfftSetStream(plan, stream));
    
    size_t worksize;
    HIPFFT_CHECK(hipfftMakePlanMany64(plan, 2, n,
                                       n, istride, idist,
                                       n, istride, idist,
                                       HIPFFT_Z2Z, batch, &worksize));
    
    printf("Reference FFT: n=[%lld,%lld] istride=%lld idist=%lld batch=%lld\n",
           n[0], n[1], istride, idist, batch);
    
    // Execute for each superbatch
    for (int k = 0; k < 2; ++k) {
      hipfftDoubleComplex* ptr = reinterpret_cast<hipfftDoubleComplex*>(d_work) + k * superbatch_offset;
      HIPFFT_CHECK(hipfftExecZ2Z(plan, ptr, ptr, HIPFFT_FORWARD));
      HIP_CHECK(hipStreamSynchronize(stream));  // Sync between superbatches
    }
    
    hipfftDestroy(plan);
    
    // Copy reference result back
    HIP_CHECK(hipMemcpy(h_reference.data(), d_work, total_size * sizeof(Complex), hipMemcpyDeviceToHost));
    printf("Reference FFT complete\n");
    
    // Print a few reference values
    printf("Reference values [0,0,0,0..4]: ");
    for (int l = 0; l < 5; ++l) {
      int idx = 0*60 + 0*20 + 0*5 + l;
      printf("(%.2f,%.2f) ", h_reference[idx].real(), h_reference[idx].imag());
    }
    printf("\n");
  }
  
  // ===== Test 0: Verify transpose round-trip is identity =====
  {
    HIP_CHECK(hipMemcpy(d_data, h_input.data(), total_size * sizeof(Complex), hipMemcpyHostToDevice));
    
    // gpuTT uses COLUMN-MAJOR ordering where dims[0] is FASTEST varying.
    // For row-major C++ data with complex (2 reals interleaved):
    //   Row-major dims: [dim0, dim1, dim2, dim3] = [2, 3, 4, 5]
    //   gpuTT column-major: [realimag=2, dim3, dim2, dim1, dim0] = [2, 5, 4, 3, 2]
    //
    // Original row-major permutation [0, 3, 1, 2] means:
    //   output[0] = input[0], output[1] = input[3], output[2] = input[1], output[3] = input[2]
    // Converting to column-major with realimag at front:
    //   Column indices: [0=realimag, 1=dim3, 2=dim2, 3=dim1, 4=dim0]
    //   We want to move dim3 (col 1) to where dim1 was (col 3), etc.
    //   Row perm [0,3,1,2] -> col_perm: for col=1..4, col corresponds to row_idx = 4-col
    //     col 1 (dim3): row_idx=3, perm_to_front[3]=2, col_perm_val = 4-2 = 2
    //     col 2 (dim2): row_idx=2, perm_to_front[2]=1, col_perm_val = 4-1 = 3
    //     col 3 (dim1): row_idx=1, perm_to_front[1]=3, col_perm_val = 4-3 = 1
    //     col 4 (dim0): row_idx=0, perm_to_front[0]=0, col_perm_val = 4-0 = 4
    //   So col_perm = [0, 2, 3, 1, 4]
    int complex_rank = ndim + 1;
    int complex_dims[5] = {2, 5, 4, 3, 2};  // Column-major: [realimag, dim3, dim2, dim1, dim0]
    int perm_fwd[5] = {0, 2, 3, 1, 4};      // Transpose forward in column-major
    // After transpose: [2, 4, 3, 5, 2] in column-major
    // Which is row-major: [2, 5, 3, 4] (reversing the middle)
    int transposed_dims[5] = {2, 4, 3, 5, 2};  // Column-major transposed
    // Inverse permutation of [0,2,3,1,4]: find where each goes
    // perm_fwd[0]=0 -> perm_bwd[0]=0
    // perm_fwd[1]=2 -> perm_bwd[2]=1
    // perm_fwd[2]=3 -> perm_bwd[3]=2
    // perm_fwd[3]=1 -> perm_bwd[1]=3
    // perm_fwd[4]=4 -> perm_bwd[4]=4
    int perm_bwd[5] = {0, 3, 1, 2, 4};
    
    gputtHandle transpose_fwd, transpose_bwd;
    GPUTT_CHECK(gputtPlan(&transpose_fwd, complex_rank, complex_dims, perm_fwd,
                          gputtDataTypeFloat64, (gputtStream)stream));
    GPUTT_CHECK(gputtPlan(&transpose_bwd, complex_rank, transposed_dims, perm_bwd,
                          gputtDataTypeFloat64, (gputtStream)stream));
    
    // Transpose forward
    GPUTT_CHECK(gputtExecute(transpose_fwd, d_data, d_temp));
    HIP_CHECK(hipStreamSynchronize(stream));
    
    // Transpose back
    GPUTT_CHECK(gputtExecute(transpose_bwd, d_temp, d_work));
    HIP_CHECK(hipStreamSynchronize(stream));
    
    gputtDestroy(transpose_fwd);
    gputtDestroy(transpose_bwd);
    
    // Check round-trip
    HIP_CHECK(hipMemcpy(h_output.data(), d_work, total_size * sizeof(Complex), hipMemcpyDeviceToHost));
    double max_err = 0.0;
    for (long long i = 0; i < total_size; ++i) {
      double err = std::abs(h_output[i] - h_input[i]);
      if (err > max_err) max_err = err;
    }
    printf("Transpose round-trip max error: %.6e\n", max_err);
    if (max_err > 1e-10) {
      printf("FAIL: Transpose round-trip failed!\n");
      // Print first few mismatches
      for (long long i = 0; i < total_size && i < 10; ++i) {
        if (std::abs(h_output[i] - h_input[i]) > 1e-10) {
          printf("  [%lld] input=(%.6f,%.6f) output=(%.6f,%.6f)\n", i,
                 h_input[i].real(), h_input[i].imag(),
                 h_output[i].real(), h_output[i].imag());
        }
      }
    } else {
      printf("Transpose round-trip OK\n");
    }
  }

  // ===== Test 1: Verify FFT on transposed layout =====
  // Compare FFT on transposed data vs reference approach
  {
    HIP_CHECK(hipMemcpy(d_data, h_input.data(), total_size * sizeof(Complex), hipMemcpyHostToDevice));
    
    int complex_rank = ndim + 1;
    int complex_dims[5] = {2, 5, 4, 3, 2};  // Column-major: [realimag, dim3, dim2, dim1, dim0]
    int perm_fwd[5] = {0, 2, 3, 1, 4};      // Transpose forward in column-major
    
    gputtHandle transpose_fwd;
    GPUTT_CHECK(gputtPlan(&transpose_fwd, complex_rank, complex_dims, perm_fwd,
                          gputtDataTypeFloat64, (gputtStream)stream));
    
    // Transpose data
    GPUTT_CHECK(gputtExecute(transpose_fwd, d_data, d_temp));
    HIP_CHECK(hipStreamSynchronize(stream));
    gputtDestroy(transpose_fwd);
    
    // Now d_temp has transposed layout [2, 5, 3, 4]
    // FFT axes [1,2] of original are now axes [2,3] of transposed (trailing)
    // FFT sizes: [3, 4]
    
    long long n_t[2] = {3, 4};
    long long batch_t = 2 * 5;  // 10 batches
    
    hipfftHandle fft_plan;
    HIPFFT_CHECK(hipfftCreate(&fft_plan));
    HIPFFT_CHECK(hipfftSetStream(fft_plan, stream));
    
    size_t worksize;
    HIPFFT_CHECK(hipfftMakePlanMany64(fft_plan, 2, n_t,
                                       n_t, 1LL, 12LL,
                                       n_t, 1LL, 12LL,
                                       HIPFFT_Z2Z, batch_t, &worksize));
    
    // Execute FFT in-place
    HIPFFT_CHECK(hipfftExecZ2Z(fft_plan,
                               reinterpret_cast<hipfftDoubleComplex*>(d_temp),
                               reinterpret_cast<hipfftDoubleComplex*>(d_temp),
                               HIPFFT_FORWARD));
    HIP_CHECK(hipStreamSynchronize(stream));
    hipfftDestroy(fft_plan);
    
    // d_temp now has FFT result in transposed layout
    printf("FFT on transposed layout executed\n");
    
    // Now let's also compute the reference in transposed layout:
    // Transpose the reference result and compare
    HIP_CHECK(hipMemcpy(d_work, h_reference.data(), total_size * sizeof(Complex), hipMemcpyHostToDevice));
    
    GPUTT_CHECK(gputtPlan(&transpose_fwd, complex_rank, complex_dims, perm_fwd,
                          gputtDataTypeFloat64, (gputtStream)stream));
    GPUTT_CHECK(gputtExecute(transpose_fwd, d_work, d_data));  // d_data = transposed reference
    HIP_CHECK(hipStreamSynchronize(stream));
    gputtDestroy(transpose_fwd);
    
    // Compare d_temp (our FFT on transposed) vs d_data (transposed reference)
    std::vector<Complex> h_test(total_size), h_ref_transposed(total_size);
    HIP_CHECK(hipMemcpy(h_test.data(), d_temp, total_size * sizeof(Complex), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_ref_transposed.data(), d_data, total_size * sizeof(Complex), hipMemcpyDeviceToHost));
    
    double max_err = 0.0;
    long long max_idx = 0;
    for (long long i = 0; i < total_size; ++i) {
      double err = std::abs(h_test[i] - h_ref_transposed[i]);
      if (err > max_err) {
        max_err = err;
        max_idx = i;
      }
    }
    printf("FFT transposed vs reference transposed max error: %.6e at index %lld\n", max_err, max_idx);
    if (max_err > 1e-10) {
      printf("  Reference: (%.6f, %.6f)\n", h_ref_transposed[max_idx].real(), h_ref_transposed[max_idx].imag());
      printf("  Test:      (%.6f, %.6f)\n", h_test[max_idx].real(), h_test[max_idx].imag());
    }
  }

  // ===== Test: transpose-FFT-transpose approach =====
  {
    // Reset d_data with original input
    HIP_CHECK(hipMemcpy(d_data, h_input.data(), total_size * sizeof(Complex), hipMemcpyHostToDevice));
    
    // Transpose to bring axes [1,2] to trailing positions
    // Original: [dim0, dim1, dim2, dim3] = [2, 3, 4, 5]
    // Target:   [dim0, dim3, dim1, dim2] = [2, 5, 3, 4]
    // Row-major permutation: [0, 3, 1, 2] (output[i] comes from input[perm[i]])
    
    // gpuTT uses COLUMN-MAJOR ordering where dims[0] is FASTEST varying.
    // Column-major: [realimag=2, dim3, dim2, dim1, dim0] = [2, 5, 4, 3, 2]
    // Column-major permutation: [0, 2, 3, 1, 4]
    int complex_rank = ndim + 1;
    int complex_dims[5] = {2, 5, 4, 3, 2};  // Column-major
    int perm_fwd[5] = {0, 2, 3, 1, 4};      // Transpose forward (column-major)
    int transposed_dims[5] = {2, 4, 3, 5, 2};  // Column-major after transpose
    int perm_bwd[5] = {0, 3, 1, 2, 4};      // Transpose back (inverse of fwd)
    
    gputtHandle transpose_fwd, transpose_bwd;
    GPUTT_CHECK(gputtPlan(&transpose_fwd, complex_rank, complex_dims, perm_fwd,
                          gputtDataTypeFloat64, (gputtStream)stream));
    GPUTT_CHECK(gputtPlan(&transpose_bwd, complex_rank, transposed_dims, perm_bwd,
                          gputtDataTypeFloat64, (gputtStream)stream));
    
    // Create FFT plan for transposed layout (axes [2,3] are now trailing)
    // n = [3, 4], istride = 1, idist = 12, batch = 10
    long long n_t[2] = {3, 4};
    long long batch_t = 2 * 5;  // product of non-FFT dims = 10
    
    hipfftHandle fft_plan;
    HIPFFT_CHECK(hipfftCreate(&fft_plan));
    HIPFFT_CHECK(hipfftSetStream(fft_plan, stream));
    
    size_t worksize;
    HIPFFT_CHECK(hipfftMakePlanMany64(fft_plan, 2, n_t,
                                       n_t, 1LL, 12LL,   // trailing axes: istride=1, idist=fft_size
                                       n_t, 1LL, 12LL,
                                       HIPFFT_Z2Z, batch_t, &worksize));
    
    // Step 1: Transpose data to bring FFT axes to trailing
    GPUTT_CHECK(gputtExecute(transpose_fwd, d_data, d_temp));
    HIP_CHECK(hipStreamSynchronize(stream));
    printf("Transpose forward complete\n");
    
    // Step 2: Execute FFT on transposed data - IN PLACE
    HIPFFT_CHECK(hipfftExecZ2Z(fft_plan, 
                               reinterpret_cast<hipfftDoubleComplex*>(d_temp),
                               reinterpret_cast<hipfftDoubleComplex*>(d_temp),  // in-place
                               HIPFFT_FORWARD));
    HIP_CHECK(hipStreamSynchronize(stream));
    printf("FFT on transposed data complete\n");
    
    // Step 3: Transpose back to original layout
    GPUTT_CHECK(gputtExecute(transpose_bwd, d_temp, d_data));
    HIP_CHECK(hipStreamSynchronize(stream));
    printf("Transpose backward complete\n");
    
    hipfftDestroy(fft_plan);
    gputtDestroy(transpose_fwd);
    gputtDestroy(transpose_bwd);
  }
  
  // Copy test result back
  HIP_CHECK(hipMemcpy(h_output.data(), d_data, total_size * sizeof(Complex), hipMemcpyDeviceToHost));
  
  // Compare results
  double max_err = 0.0;
  long long max_idx = 0;
  for (long long i = 0; i < total_size; ++i) {
    double err = std::abs(h_output[i] - h_reference[i]);
    if (err > max_err) {
      max_err = err;
      max_idx = i;
    }
  }
  
  printf("\nMax error: %.6e at index %lld\n", max_err, max_idx);
  printf("  Reference: (%.6f, %.6f)\n", h_reference[max_idx].real(), h_reference[max_idx].imag());
  printf("  Test:      (%.6f, %.6f)\n", h_output[max_idx].real(), h_output[max_idx].imag());
  
  // Cleanup
  HIP_CHECK(hipFree(d_data));
  HIP_CHECK(hipFree(d_work));
  HIP_CHECK(hipFree(d_temp));
  HIP_CHECK(hipStreamDestroy(stream));
  
  MPI_Finalize();
  
  if (max_err < 1e-10) {
    printf("\nPASS\n");
    return 0;
  } else {
    printf("\nFAIL\n");
    return 1;
  }
}
