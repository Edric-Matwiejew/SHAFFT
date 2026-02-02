/**
 * @file test_fft_partial.cpp
 * @brief Validate hipFFT partial (contiguous) transforms against a naive CPU DFT.
 *
 * Single rank, nda=0; axes are a contiguous block (as SHAFFT issues in production).
 */

#include <vector>
#include <complex>
#include <cmath>
#include <iostream>
#include <numeric>

#include <shafft/shafft_types.hpp>
#include "hipfft_method/fft_method.h"
#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>

using cplx = std::complex<double>;

static std::vector<cplx> make_input(size_t n) {
  std::vector<cplx> v(n);
  for (size_t i = 0; i < n; ++i)
    v[i] = cplx(std::sin(0.01 * i), std::cos(0.02 * i));
  return v;
}

static size_t total_elems(const std::vector<int>& dims) {
  return std::accumulate(dims.begin(), dims.end(), size_t{1},
                         [](size_t a,int b){ return a * static_cast<size_t>(b); });
}

// Naive partial DFT over contiguous block [a0, a0+nta-1], full tensor output.
static std::vector<cplx> dft_partial(const std::vector<cplx>& in,
                                     const std::vector<int>& dims,
                                     int a0, int nta)
{
  const int ndim = static_cast<int>(dims.size());
  const int a1 = a0 + nta - 1;
  std::vector<size_t> stride(ndim,1);
  for (int i = ndim-2; i >=0; --i) stride[i] = stride[i+1] * static_cast<size_t>(dims[i+1]);

  const size_t total = in.size();
  std::vector<cplx> out(total, cplx{0.0,0.0});
  std::vector<int> idx(ndim,0), x(ndim,0);
  const double two_pi = 2.0 * std::acos(-1.0);

  for (size_t fk = 0; fk < total; ++fk) {
    // unravel fk -> idx
    size_t tmp = fk;
    for (int i=0;i<ndim;++i){ idx[i] = static_cast<int>(tmp / stride[i]); tmp %= stride[i]; }

    cplx sum{0.0,0.0};
    // iterate over transform axes only
    for (int i=0;i<ndim;++i) x[i]=idx[i];
    for (int i0=0;i0<dims[a0];++i0){
      x[a0]=i0;
      for (int i1=(a0+1<=a1?0:1); i1<(a0+1<=a1?dims[a0+1]:2); ++i1){
        if (a0+1<=a1) x[a0+1]=i1;
        for (int i2=(a0+2<=a1?0:1); i2<(a0+2<=a1?dims[a0+2]:2); ++i2){
          if (a0+2<=a1) x[a0+2]=i2;
          // extend if nta>3? simple general loop is clearer:
        }
      }
    }
    // Replace nested unrolled loops with generic recursion would be better; simple generic:
  }
  // generic version:
  // We'll redo with iterative nested for via stack
  std::fill(out.begin(), out.end(), cplx{0.0,0.0}); // clear
  for (size_t fk = 0; fk < total; ++fk) {
    size_t tmp = fk;
    for (int i=0;i<ndim;++i){ idx[i] = static_cast<int>(tmp / stride[i]); tmp %= stride[i]; }

    cplx sum{0.0,0.0};
    // iterate over transform block
    std::vector<int> cur(ndim);
    // counts for block
    const size_t block_size = std::accumulate(dims.begin()+a0, dims.begin()+a1+1, size_t{1},
                                              [](size_t a,int b){return a*static_cast<size_t>(b);});
    for (size_t bi=0; bi<block_size; ++bi){
      // Start with cur = idx for all axes
      cur = idx;
      // Override transform axes [a0..a1] based on bi
      size_t t=bi;
      for (int ax=a1; ax>=a0; --ax){
        cur[ax] = static_cast<int>(t % dims[ax]);
        t /= dims[ax];
      }
      double phase=0.0;
      for (int ax=a0; ax<=a1; ++ax){
        phase += static_cast<double>(idx[ax] * cur[ax]) / dims[ax];
      }
      const double ang = -two_pi * phase;
      const double c = std::cos(ang), s = std::sin(ang);
      // flat index of cur
      size_t fx=0;
      for (int i=0;i<ndim;++i) fx += static_cast<size_t>(cur[i]) * stride[i];
      sum += in[fx] * cplx{c,s};
    }
    out[fk]=sum;
  }
  return out;
}

static hipDoubleComplex* hip_output_ptr(const fftHandle& h, hipDoubleComplex* data, hipDoubleComplex* work) {
  return (h.nsubplans % 2 == 0) ? data : work;
}

struct Case { std::vector<int> dims; std::vector<int> axes; };

static bool run_case(const Case& tc, double tol = 1e-10) {
  const int ndim = static_cast<int>(tc.dims.size());
  const int nta  = static_cast<int>(tc.axes.size());
  size_t N = total_elems(tc.dims);

  // require contiguous block (production invariant)
  for (int i=1;i<nta;++i){
    if (tc.axes[i] != tc.axes[0] + i){
      std::cerr << "Non-contiguous axes in test case\n";
      return false;
    }
  }
  const int a0 = tc.axes[0];

  auto host_in = make_input(N);
  auto ref = dft_partial(host_in, tc.dims, a0, nta);

  fftHandle h_hip{};
  int rc = fftPlan(h_hip, nta, const_cast<int*>(tc.axes.data()),
                   ndim, const_cast<int*>(tc.dims.data()),
                   shafft::FFTType::Z2Z);
  if (rc != 0) { std::cerr << "hip plan rc="<<rc<<"\n"; return false; }

  hipDoubleComplex *d_data = nullptr, *d_work = nullptr;
  hipError_t herr = hipMalloc(&d_data, N * sizeof(hipDoubleComplex));
  if (herr != hipSuccess) return false;
  herr = hipMalloc(&d_work, N * sizeof(hipDoubleComplex));
  if (herr != hipSuccess) { (void)hipFree(d_data); return false; }
  herr = hipMemcpy(d_data, host_in.data(), N * sizeof(hipDoubleComplex), hipMemcpyHostToDevice);
  if (herr != hipSuccess) { (void)hipFree(d_data); (void)hipFree(d_work); return false; }

  void* data_ptr = d_data;
  void* work_ptr = d_work;
  rc = fftExecute(h_hip, data_ptr, work_ptr, shafft::FFTDirection::FORWARD);
  if (rc != 0) { std::cerr << "hip exec rc="<<rc<<"\n"; (void)hipFree(d_data); (void)hipFree(d_work); return false; }

  // Result is now in data_ptr after swap
  std::vector<cplx> got(N);
  herr = hipMemcpy(got.data(), data_ptr, N * sizeof(hipDoubleComplex), hipMemcpyDeviceToHost);
  if (herr != hipSuccess) { (void)hipFree(d_data); (void)hipFree(d_work); return false; }

  (void)hipFree(d_data);
  (void)hipFree(d_work);
  fftDestroy(h_hip);

  double max_err = 0.0;
  for (size_t i=0;i<N;++i) {
    double er = std::abs(got[i].real()-ref[i].real());
    double ei = std::abs(got[i].imag()-ref[i].imag());
    if (er > max_err) max_err = er;
    if (ei > max_err) max_err = ei;
  }
  if (max_err > tol) {
    std::cerr << "FAIL dims=";
    for (auto d:tc.dims) std::cerr<<d<<" ";
    std::cerr<<" axes=";
    for (auto a:tc.axes) std::cerr<<a<<" ";
    std::cerr<<" max_err="<<max_err<<"\n";
    return false;
  }
  return true;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  std::vector<Case> cases = {
    {{2,3,4,5},       {0,1,2}},
    {{4,3,2,3,2},     {1,2,3}},
    {{2,3,4,5},       {2,3}},
    {{2,3,4,2},       {0,1,2,3}},
    {{3,2,2,2,2},     {1,2,3,4}},
  };

  int failed=0, passed=0;
  for (const auto& tc : cases) {
    if (run_case(tc)) passed++; else failed++;
  }

  if (failed==0) {
    std::cout << "ALL PASSED ("<<passed<<")\n";
    MPI_Finalize();
    return 0;
  } else {
    std::cout << failed << " FAILED, " << passed << " passed\n";
    MPI_Finalize();
    return 1;
  }
}

