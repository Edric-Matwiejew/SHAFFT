/**
 * @file test_superbatch_threshold.cpp
 * @brief Unit tests for per-FFT-rank superbatch thresholds.
 *
 * Verifies that:
 *  - Default thresholds are correct (1D=16, 2D=16, 3D=1)
 *  - getSuperbatchThreshold() returns correct values per rank
 *  - Plans select the correct FFTMethod based on superbatch count and FFT rank
 *
 * This test requires the hipFFT backend.
 */
#include "test_utils.hpp"
#include <mpi.h>

#include <cstdlib>
#include <iostream>
#include <vector>

#if SHAFFT_BACKEND_HIPFFT
#include "fftnd_method.hpp"
#include "nd/transpose_fftnd.hpp"

#include <hip/hip_runtime.h>
#endif

#if !SHAFFT_BACKEND_HIPFFT
int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  // hipFFT backend not built; silently skip to keep CTest green on CPU builds.
  std::cout << "test_superbatch_threshold: SKIPPED (hipFFT backend not built)\n";
  return 0;
}
#else

static const char* method_to_string(FFTMethod m) {
  switch (m) {
  case FFTMethod::STRIDED:
    return "STRIDED";
  case FFTMethod::TRANSPOSE:
    return "TRANSPOSE";
  case FFTMethod::TRAILING:
    return "TRAILING";
  default:
    return "UNKNOWN";
  }
}

// Test: Default thresholds are correct
static bool test_default_thresholds() {
  // Reset cache to ensure we're testing fresh state (even if other tests ran first)
  resetSuperbatchThresholdCache();

  // Clear any environment overrides for this test
  unsetenv("SHAFFT_SUPERBATCH_THRESHOLD");
  unsetenv("SHAFFT_SUPERBATCH_THRESHOLD_1D");
  unsetenv("SHAFFT_SUPERBATCH_THRESHOLD_2D");
  unsetenv("SHAFFT_SUPERBATCH_THRESHOLD_3D");

  long long t1 = getSuperbatchThreshold(1);
  long long t2 = getSuperbatchThreshold(2);
  long long t3 = getSuperbatchThreshold(3);

  std::cout << "    1D threshold: " << t1 << " (expected: 16)\n";
  std::cout << "    2D threshold: " << t2 << " (expected: 16)\n";
  std::cout << "    3D threshold: " << t3 << " (expected: 1)\n";

  // Default thresholds based on MI250X benchmarks
  bool ok = true;
  if (t1 != 16) {
    std::cerr << "    ERROR: 1D threshold=" << t1 << ", expected 16\n";
    ok = false;
  }
  if (t2 != 16) {
    std::cerr << "    ERROR: 2D threshold=" << t2 << ", expected 16\n";
    ok = false;
  }
  if (t3 != 1) {
    std::cerr << "    ERROR: 3D threshold=" << t3 << ", expected 1\n";
    ok = false;
  }

  return ok;
}

// Test: 3D FFT with superbatch > 1 uses TRANSPOSE (since threshold=1)
static bool test_fft3d_uses_transpose() {
  // Create a 3D FFT with leading batch dimension AND trailing suffix (superbatch=2, suffix=2)
  // Tensor layout: [2, 4, 4, 4, 2] with FFT on axes [1,2,3]
  // Since 3D threshold=1 and superbatch=2 > 1, should use TRANSPOSE
  std::vector<int> dims = {2, 4, 4, 4, 2};
  std::vector<int> axes = {1, 2, 3};

  FFTNDHandle plan;
  int rc = fftndPlan(plan,
                     static_cast<int>(axes.size()),
                     axes.data(),
                     static_cast<int>(dims.size()),
                     dims.data(),
                     shafft::FFTType::Z2Z,
                     nullptr,
                     nullptr);

  if (rc != 0) {
    std::cerr << "    fftPlan failed with rc=" << rc << "\n";
    return false;
  }

  bool ok = true;

  // Should have 1 subplan for the 3D FFT
  if (plan.nsubplans != 1) {
    std::cerr << "    Expected 1 subplan, got " << plan.nsubplans << "\n";
    ok = false;
  }

  // Check that the method is TRANSPOSE (since superbatch=2 > threshold=1)
  if (plan.methods[0] != FFTMethod::TRANSPOSE) {
    std::cerr << "    Expected TRANSPOSE, got " << method_to_string(plan.methods[0]) << "\n";
    ok = false;
  } else {
    std::cout << "    3D FFT with superbatch=2, suffix=2: " << method_to_string(plan.methods[0])
              << "\n";
  }

  fftndDestroy(plan);
  return ok;
}

// Test: 2D FFT with small superbatch uses STRIDED (below threshold)
static bool test_fft2d_small_superbatch_uses_strided() {
  // Create a 2D FFT with superbatch=4 (below threshold=16)
  // Tensor layout: [4, 8, 8, 2] with FFT on axes [1,2]
  // Since superbatch=4 < threshold=16, should use STRIDED
  std::vector<int> dims = {4, 8, 8, 2};
  std::vector<int> axes = {1, 2};

  FFTNDHandle plan;
  int rc = fftndPlan(plan,
                     static_cast<int>(axes.size()),
                     axes.data(),
                     static_cast<int>(dims.size()),
                     dims.data(),
                     shafft::FFTType::Z2Z,
                     nullptr,
                     nullptr);

  if (rc != 0) {
    std::cerr << "    fftPlan failed with rc=" << rc << "\n";
    return false;
  }

  bool ok = true;

  if (plan.nsubplans != 1) {
    std::cerr << "    Expected 1 subplan, got " << plan.nsubplans << "\n";
    ok = false;
  }

  // Check that the method is STRIDED (since superbatch=4 < threshold=16)
  if (plan.methods[0] != FFTMethod::STRIDED) {
    std::cerr << "    Expected STRIDED for superbatch=4 < threshold=16, got "
              << method_to_string(plan.methods[0]) << "\n";
    ok = false;
  } else {
    std::cout << "    2D FFT with superbatch=4: " << method_to_string(plan.methods[0]) << "\n";
  }

  fftndDestroy(plan);
  return ok;
}

// Test: 2D FFT with large superbatch uses TRANSPOSE (above threshold)
static bool test_fft2d_large_superbatch_uses_transpose() {
  // Create a 2D FFT with superbatch=32 (above threshold=16)
  // Tensor layout: [32, 8, 8, 2] with FFT on axes [1,2]
  // Since superbatch=32 > threshold=16, should use TRANSPOSE
  std::vector<int> dims = {32, 8, 8, 2};
  std::vector<int> axes = {1, 2};

  FFTNDHandle plan;
  int rc = fftndPlan(plan,
                     static_cast<int>(axes.size()),
                     axes.data(),
                     static_cast<int>(dims.size()),
                     dims.data(),
                     shafft::FFTType::Z2Z,
                     nullptr,
                     nullptr);

  if (rc != 0) {
    std::cerr << "    fftPlan failed with rc=" << rc << "\n";
    return false;
  }

  bool ok = true;

  if (plan.nsubplans != 1) {
    std::cerr << "    Expected 1 subplan, got " << plan.nsubplans << "\n";
    ok = false;
  }

  // Check that the method is TRANSPOSE (since superbatch=32 > threshold=16)
  if (plan.methods[0] != FFTMethod::TRANSPOSE) {
    std::cerr << "    Expected TRANSPOSE for superbatch=32 > threshold=16, got "
              << method_to_string(plan.methods[0]) << "\n";
    ok = false;
  } else {
    std::cout << "    2D FFT with superbatch=32: " << method_to_string(plan.methods[0]) << "\n";
  }

  fftndDestroy(plan);
  return ok;
}

// Test: Trailing FFT uses TRAILING method (no superbatch optimization needed)
static bool test_trailing_fft_uses_trailing_method() {
  // Create a 2D FFT on trailing axes (no superbatch)
  // Tensor layout: [4, 8, 8] with FFT on axes [1,2] (trailing)
  std::vector<int> dims = {4, 8, 8};
  std::vector<int> axes = {1, 2};

  FFTNDHandle plan;
  int rc = fftndPlan(plan,
                     static_cast<int>(axes.size()),
                     axes.data(),
                     static_cast<int>(dims.size()),
                     dims.data(),
                     shafft::FFTType::Z2Z,
                     nullptr,
                     nullptr);

  if (rc != 0) {
    std::cerr << "    fftPlan failed with rc=" << rc << "\n";
    return false;
  }

  bool ok = true;

  if (plan.nsubplans != 1) {
    std::cerr << "    Expected 1 subplan, got " << plan.nsubplans << "\n";
    ok = false;
  }

  // Check that the method is TRAILING (FFT axes are trailing, batched natively)
  if (plan.methods[0] != FFTMethod::TRAILING) {
    std::cerr << "    Expected TRAILING for trailing axes, got "
              << method_to_string(plan.methods[0]) << "\n";
    ok = false;
  } else {
    std::cout << "    2D FFT on trailing axes: " << method_to_string(plan.methods[0]) << "\n";
  }

  fftndDestroy(plan);
  return ok;
}

// Test: 1D FFT with superbatch at threshold boundary
static bool test_fft1d_threshold_boundary() {
  // Create 1D FFT with superbatch=16 (at threshold)
  // superbatch > threshold triggers TRANSPOSE, so 16 > 16 is false -> STRIDED
  // superbatch=17 > 16 -> TRANSPOSE
  std::vector<int> dims_at = {16, 64, 4};    // superbatch=16, FFT on axis 1
  std::vector<int> dims_above = {17, 64, 4}; // superbatch=17, FFT on axis 1
  std::vector<int> axes = {1};

  FFTNDHandle plan_at, plan_above;

  int rc1 = fftndPlan(plan_at,
                      static_cast<int>(axes.size()),
                      axes.data(),
                      static_cast<int>(dims_at.size()),
                      dims_at.data(),
                      shafft::FFTType::Z2Z,
                      nullptr,
                      nullptr);
  int rc2 = fftndPlan(plan_above,
                      static_cast<int>(axes.size()),
                      axes.data(),
                      static_cast<int>(dims_above.size()),
                      dims_above.data(),
                      shafft::FFTType::Z2Z,
                      nullptr,
                      nullptr);

  if (rc1 != 0 || rc2 != 0) {
    std::cerr << "    fftPlan failed\n";
    return false;
  }

  bool ok = true;

  // At threshold (16): should be STRIDED (16 > 16 is false)
  std::cout << "    1D FFT superbatch=16 (at threshold): " << method_to_string(plan_at.methods[0])
            << "\n";
  if (plan_at.methods[0] != FFTMethod::STRIDED) {
    std::cerr << "    Expected STRIDED at threshold, got " << method_to_string(plan_at.methods[0])
              << "\n";
    ok = false;
  }

  // Above threshold (17): should be TRANSPOSE (17 > 16 is true)
  std::cout << "    1D FFT superbatch=17 (above threshold): "
            << method_to_string(plan_above.methods[0]) << "\n";
  if (plan_above.methods[0] != FFTMethod::TRANSPOSE) {
    std::cerr << "    Expected TRANSPOSE above threshold, got "
              << method_to_string(plan_above.methods[0]) << "\n";
    ok = false;
  }

  fftndDestroy(plan_at);
  fftndDestroy(plan_above);
  return ok;
}

int main(int argc, char** argv) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::fprintf(stderr, "MPI_Init failed\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  test::TestRunner runner("Superbatch Threshold Unit Tests");
  runner.run("default_thresholds", test_default_thresholds);
  runner.run("fft3d_uses_transpose", test_fft3d_uses_transpose);
  runner.run("fft2d_small_superbatch_uses_strided", test_fft2d_small_superbatch_uses_strided);
  runner.run("fft2d_large_superbatch_uses_transpose", test_fft2d_large_superbatch_uses_transpose);
  runner.run("trailing_fft_uses_trailing_method", test_trailing_fft_uses_trailing_method);
  runner.run("fft1d_threshold_boundary", test_fft1d_threshold_boundary);
  int ret = runner.finalize();

  MPI_Finalize();
  return ret;
}

#endif // SHAFFT_BACKEND_HIPFFT
