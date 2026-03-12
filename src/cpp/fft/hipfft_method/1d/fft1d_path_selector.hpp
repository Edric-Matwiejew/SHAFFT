// fft1d_path_selector.hpp
// Path selection heuristic for distributed 1D FFT.
// - Path A: Cooley-Tukey P×P (fast) - when N' % P² == 0 with acceptable padding
// - Path B: Bluestein fallback (exact for any N)

#ifndef SHAFFT_FFT1D_PATH_SELECTOR_HPP
#define SHAFFT_FFT1D_PATH_SELECTOR_HPP

#include <cstddef>

namespace shafft {

/// Execution path for distributed 1D FFT
enum class FFT1DPath {
  COOLEY_TUKEY, // Path A: N' must be divisible by P²
  BLUESTEIN     // Path B: Works for any N
};

/// Result of path selection
struct FFT1DPathConfig {
  FFT1DPath path;
  size_t fftLength;    // N' (Path A) or M (Path B)
  size_t localSize;    // L = fftLength / P
  size_t chunk;        // C = L / P (only valid for Path A, 0 for Path B)
  double paddingRatio; // fftLength / N
};

/// Compute padded size N' such that N' % P² == 0 (for Path A)
inline size_t computePaddedSizeP2(size_t N, size_t P) {
  size_t P2 = P * P;
  return ((N + P2 - 1) / P2) * P2;
}

/// Compute convolution length M for Bluestein such that M >= 2N-1 and M % P² == 0
inline size_t computeBluesteinM(size_t N, size_t P) {
  size_t M0 = 2 * N - 1;
  size_t P2 = P * P;
  return ((M0 + P2 - 1) / P2) * P2;
}

/// Select the execution path based on N, P, and tolerance tau.
///
/// @param N         True problem size (user's globalN)
/// @param P         Number of MPI ranks
/// @param tau       Padding tolerance (default 1.25 = 25% overhead)
/// @param allowPadding  If false and Path A would pad, force Path B
/// @return Configuration specifying which path to use and derived sizes
inline FFT1DPathConfig selectPath(size_t N, size_t P, double tau = 1.25, bool allowPadding = true) {
  FFT1DPathConfig config;

  // Compute Path A parameters
  size_t Nprime = computePaddedSizeP2(N, P);
  double ratioA = static_cast<double>(Nprime) / static_cast<double>(N);

  // Compute Path B parameters
  size_t M = computeBluesteinM(N, P);
  double ratioB = static_cast<double>(M) / static_cast<double>(N);

  // Decision logic:
  // 1. If N is already divisible by P² (no padding), always use Path A
  // 2. If allowPadding == false and Path A would pad, use Path B
  // 3. If Path A padding ratio <= tau, use Path A
  // 4. Otherwise use Path B

  bool usePathA = false;

  if (Nprime == N) {
    // Perfect fit for Path A - no padding needed
    usePathA = true;
  } else if (!allowPadding) {
    // User forbids padding semantics change - must use Bluestein
    usePathA = false;
  } else if (ratioA <= tau) {
    // Path A padding is within tolerance
    usePathA = true;
  } else {
    // Path A padding exceeds tolerance - use Bluestein
    usePathA = false;
  }

  if (usePathA) {
    config.path = FFT1DPath::COOLEY_TUKEY;
    config.fftLength = Nprime;
    config.localSize = Nprime / P;
    config.chunk = config.localSize / P;
    config.paddingRatio = ratioA;
  } else {
    config.path = FFT1DPath::BLUESTEIN;
    config.fftLength = M;
    config.localSize = M / P;
    config.chunk = 0; // Not used for Bluestein outer interface
    config.paddingRatio = ratioB;
  }

  return config;
}

/// Get a string description of the path for diagnostics
inline const char* pathToString(FFT1DPath path) {
  switch (path) {
  case FFT1DPath::COOLEY_TUKEY:
    return "Cooley-Tukey";
  case FFT1DPath::BLUESTEIN:
    return "Bluestein";
  default:
    return "Unknown";
  }
}

} // namespace shafft

#endif // SHAFFT_FFT1D_PATH_SELECTOR_HPP
