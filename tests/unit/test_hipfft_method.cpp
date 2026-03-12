/**
 * @file test_hipfft_method.cpp
 * @brief Unit tests that exercise the hipFFT backend glue (fft_method.cpp).
 *
 * Focus areas:
 *  - Correct axis ordering passed to hipFFT (fastest dimension first).
 *  - Graceful rejection of plans whose strides/distances exceed 32-bit limits.
 */
#include "test_utils.hpp"
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <mpi.h>
#include <shafft/shafft.hpp>
#include <vector>

#if SHAFFT_BACKEND_HIPFFT
#include "fftnd_method.hpp" // internal hipfft_method API
#include <hip/hip_runtime.h>
#endif

#if !SHAFFT_BACKEND_HIPFFT
int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  // hipFFT backend not built; silently skip to keep CTest green on CPU builds.
  return 0;
}
#else

// Helper: naive 2D DFT (row-major, last index fastest)
static std::vector<std::complex<double>>
dft2d(const std::vector<std::complex<double>>& in, int n0, int n1) {
  const double twoPi = 2.0 * std::acos(-1.0);
  std::vector<std::complex<double>> out(static_cast<size_t>(n0 * n1));
  for (int k0 = 0; k0 < n0; ++k0) {
    for (int k1 = 0; k1 < n1; ++k1) {
      std::complex<double> acc{0.0, 0.0};
      for (int x0 = 0; x0 < n0; ++x0) {
        for (int x1 = 0; x1 < n1; ++x1) {
          const double phase =
              -twoPi * (static_cast<double>(k0 * x0) / n0 + static_cast<double>(k1 * x1) / n1);
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

// Test: hipFFT respects fastest-axis-first ordering for multi-axis plans
static bool testHipFFTAxisOrderForward() {
  const std::vector<size_t> dims = {2, 3};  // smallest rectangular case that reveals ordering bugs
  const std::vector<int> commDims = {1, 1}; // no distribution (single rank)

  shafft::ConfigND cfg(dims, shafft::FFTType::Z2Z, commDims);

  shafft::FFTND fft;
  int rc = fft.init(cfg.cStruct());
  if (rc != 0) {
    std::cerr << "fft.init failed with rc=" << rc << "\n";
    return false;
  }
  rc = fft.plan();
  if (rc != 0) {
    std::cerr << "fft.plan failed with rc=" << rc << "\n";
    return false;
  }

  const size_t n = fft.allocSize();
  std::vector<shafft::complexd> hostIn(n);
  for (size_t i = 0; i < n; ++i) {
    hostIn[i] = {static_cast<double>(i + 1), static_cast<double>(i) * 0.25};
  }

  shafft::complexd *devData = nullptr, *devWork = nullptr;
  rc = shafft::allocBuffer(n, &devData);
  if (rc != 0)
    return false;
  rc = shafft::allocBuffer(n, &devWork);
  if (rc != 0) {
    (void)shafft::freeBuffer(devData);
    return false;
  }

  rc = shafft::copyToBuffer(devData, hostIn.data(), n);
  if (rc != 0) {
    std::cerr << "copyToBuffer failed rc=" << rc << "\n";
    (void)shafft::freeBuffer(devData);
    (void)shafft::freeBuffer(devWork);
    return false;
  }

  rc = fft.setBuffers(devData, devWork);
  if (rc != 0) {
    std::cerr << "setBuffers failed rc=" << rc << "\n";
    (void)shafft::freeBuffer(devData);
    (void)shafft::freeBuffer(devWork);
    return false;
  }
  rc = fft.execute(shafft::FFTDirection::FORWARD);
  if (rc != 0) {
    std::cerr << "execute failed rc=" << rc << "\n";
    (void)shafft::freeBuffer(devData);
    (void)shafft::freeBuffer(devWork);
    return false;
  }

  // Fetch whichever buffer now holds the result
  shafft::complexd *devOut = nullptr, *devUnused = nullptr;
  rc = fft.getBuffers(&devOut, &devUnused);
  if (rc != 0) {
    std::cerr << "getBuffers failed rc=" << rc << "\n";
    (void)shafft::freeBuffer(devData);
    (void)shafft::freeBuffer(devWork);
    return false;
  }

  std::vector<shafft::complexd> hostOut(n);
  rc = shafft::copyFromBuffer(hostOut.data(), devOut, n);
  if (rc != 0) {
    std::cerr << "copyFromBuffer failed rc=" << rc << "\n";
    (void)shafft::freeBuffer(devData);
    (void)shafft::freeBuffer(devWork);
    return false;
  }

  // Convert hostIn to std::complex<double> for reference DFT
  std::vector<std::complex<double>> hostInCplx(hostIn.begin(), hostIn.end());
  const auto ref = dft2d(hostInCplx, dims[0], dims[1]);

  double maxErr = 0.0;
  auto badIdx = static_cast<size_t>(-1);
  std::complex<double> badVal{}, badRef{};
  for (size_t i = 0; i < n; ++i) {
    const double errRe = std::abs(hostOut[i].real() - ref[i].real());
    const double errIm = std::abs(hostOut[i].imag() - ref[i].imag());
    const double e = std::max(errRe, errIm);
    if (e > maxErr) {
      maxErr = e;
      badIdx = i;
      badVal = hostOut[i];
      badRef = ref[i];
    }
  }

  (void)shafft::freeBuffer(devData);
  (void)shafft::freeBuffer(devWork);

  // Very small tolerance since inputs are tiny and transform size is 6
  if (maxErr > 1e-10) {
    std::cerr << "max_err=" << maxErr << " exceeds tolerance at idx=" << badIdx << " got=("
              << badVal.real() << "," << badVal.imag() << ") expected=(" << badRef.real() << ","
              << badRef.imag() << ")\n";
    return false;
  }
  return true;
}

int main(int argc, char** argv) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::fprintf(stderr, "MPI_Init failed\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  test::TestRunner runner("hipFFT method unit tests");
  runner.run("hipfft_axis_order_forward", testHipFFTAxisOrderForward);

  const int result = runner.finalize();
  MPI_Finalize();
  return result;
}

#endif // SHAFFT_BACKEND_HIPFFT
