/**
 * @file test_hipfft_method.cpp
 * @brief Unit tests that exercise the hipFFT backend glue (fft_method.cpp).
 *
 * Focus areas:
 *  - Correct axis ordering passed to hipFFT (fastest dimension first).
 *  - Graceful rejection of plans whose strides/distances exceed 32-bit limits.
 */

#include <shafft/shafft.hpp>
#include <mpi.h>
#include <complex>
#include <vector>
#include <cmath>
#include <iostream>

#if SHAFFT_BACKEND_HIPFFT
#include "fft_method.h"  // internal hipfft_method API
#include <hip/hip_runtime.h>
#endif

#if !SHAFFT_BACKEND_HIPFFT
int main(int argc, char** argv) {
  (void)argc; (void)argv;
  // hipFFT backend not built; silently skip to keep CTest green on CPU builds.
  return 0;
}
#else

// Simple test harness -------------------------------------------------------
static int g_passed = 0;
static int g_failed = 0;

#define TEST(name) \
  static bool test_##name(); \
  static bool run_##name() { \
    bool ok = test_##name(); \
    if (ok) { g_passed++; std::cout << "  " #name " PASS\n"; } \
    else    { g_failed++; std::cout << "  " #name " FAIL\n"; } \
    return ok; \
  } \
  static bool test_##name()

//------------------------------------------------------------------------------
// Helper: naive 2D DFT (row-major, last index fastest)
//------------------------------------------------------------------------------
static std::vector<std::complex<double>> dft2d(const std::vector<std::complex<double>>& in,
                                               int n0, int n1)
{
  const double two_pi = 2.0 * std::acos(-1.0);
  std::vector<std::complex<double>> out(n0 * n1);
  for (int k0 = 0; k0 < n0; ++k0) {
    for (int k1 = 0; k1 < n1; ++k1) {
      std::complex<double> acc{0.0, 0.0};
      for (int x0 = 0; x0 < n0; ++x0) {
        for (int x1 = 0; x1 < n1; ++x1) {
          const double phase = -two_pi * (static_cast<double>(k0 * x0) / n0
                                        + static_cast<double>(k1 * x1) / n1);
          const double c = std::cos(phase);
          const double s = std::sin(phase);
          acc += in[x0 * n1 + x1] * std::complex<double>(c, s);
        }
      }
      out[k0 * n1 + k1] = acc;
    }
  }
  return out;
}

//------------------------------------------------------------------------------
// Test: hipFFT respects fastest-axis-first ordering for multi-axis plans
//------------------------------------------------------------------------------
TEST(hipfft_axis_order_forward) {
  const std::vector<int> dims = {2, 3}; // smallest rectangular case that reveals ordering bugs

  shafft::Plan plan;
  int rc = plan.init(/*nda=*/0, dims, shafft::FFTType::Z2Z, MPI_COMM_WORLD);
  if (rc != 0) {
    std::cerr << "plan.init failed with rc=" << rc << "\n";
    return false;
  }

  const size_t n = plan.allocSize();
  std::vector<shafft::complexd> host_in(n);
  for (size_t i = 0; i < n; ++i) {
    host_in[i] = {static_cast<double>(i + 1), static_cast<double>(i) * 0.25};
  }

  shafft::complexd *dev_data = nullptr, *dev_work = nullptr;
  rc = shafft::allocBuffer(n, &dev_data); if (rc != 0) return false;
  rc = shafft::allocBuffer(n, &dev_work); if (rc != 0) { (void)shafft::freeBuffer(dev_data); return false; }

  rc = shafft::copyToBuffer(dev_data, host_in.data(), n);
  if (rc != 0) { std::cerr << "copyToBuffer failed rc=" << rc << "\n"; return false; }

  rc = plan.setBuffers(dev_data, dev_work);
  if (rc != 0) { std::cerr << "setBuffers failed rc=" << rc << "\n"; return false; }
  rc = plan.execute(shafft::FFTDirection::FORWARD);
  if (rc != 0) { std::cerr << "execute failed rc=" << rc << "\n"; return false; }

  // Fetch whichever buffer now holds the result
  shafft::complexd *dev_out = nullptr, *dev_unused = nullptr;
  rc = plan.getBuffers(&dev_out, &dev_unused);
  if (rc != 0) { std::cerr << "getBuffers failed rc=" << rc << "\n"; return false; }

  std::vector<shafft::complexd> host_out(n);
  rc = shafft::copyFromBuffer(host_out.data(), dev_out, n);
  if (rc != 0) { std::cerr << "copyFromBuffer failed rc=" << rc << "\n"; return false; }

  const auto ref = dft2d({host_in.begin(), host_in.end()}, dims[0], dims[1]);

  double max_err = 0.0;
  size_t bad_idx = static_cast<size_t>(-1);
  std::complex<double> bad_val{}, bad_ref{};
  for (size_t i = 0; i < n; ++i) {
    const double err_re = std::abs(host_out[i].real() - ref[i].real());
    const double err_im = std::abs(host_out[i].imag() - ref[i].imag());
    const double e = std::max(err_re, err_im);
    if (e > max_err) {
      max_err = e;
      bad_idx = i;
      bad_val = host_out[i];
      bad_ref = ref[i];
    }
  }

  (void)shafft::freeBuffer(dev_data);
  (void)shafft::freeBuffer(dev_work);

  // Very small tolerance since inputs are tiny and transform size is 6
  if (max_err > 1e-10) {
    std::cerr << "max_err=" << max_err << " exceeds tolerance at idx=" << bad_idx
              << " got=(" << bad_val.real() << "," << bad_val.imag()
              << ") expected=(" << bad_ref.real() << "," << bad_ref.imag() << ")\n";
    return false;
  }
  return true;
}

//------------------------------------------------------------------------------
// Test: plan creation rejects dimensions that overflow 32-bit hipFFT parameters
//------------------------------------------------------------------------------
TEST(hipfft_rejects_int_overflow) {
  // Product exceeds INT_MAX, so transform_parameters should fail
  const std::vector<int> dims = {65536, 65536};
  shafft::Plan plan;
  int rc = plan.init(/*nda=*/0, dims, shafft::FFTType::Z2Z, MPI_COMM_WORLD);
  return rc != 0;
}

//------------------------------------------------------------------------------
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  std::cout << "hipFFT method unit tests" << std::endl;
  run_hipfft_axis_order_forward();
  run_hipfft_rejects_int_overflow();

  if (g_failed == 0) {
    std::cout << "ALL PASSED (" << g_passed << ")" << std::endl;
  } else {
    std::cout << g_failed << " FAILED, " << g_passed << " passed" << std::endl;
  }

  MPI_Finalize();
  return g_failed == 0 ? 0 : 1;
}

#endif // SHAFFT_BACKEND_HIPFFT
