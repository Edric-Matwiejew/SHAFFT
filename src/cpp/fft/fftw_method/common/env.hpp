// env.hpp - FFTW environment variable helpers
//
// Parses environment variables for FFTW configuration:
//   SHAFFT_FFTW_THREADS  - Number of threads (default: 1)
//   SHAFFT_FFTW_PLANNER  - Planner strategy: ESTIMATE, MEASURE, PATIENT, EXHAUSTIVE

#ifndef SHAFFT_FFT_FFTW_COMMON_ENV_HPP
#define SHAFFT_FFT_FFTW_COMMON_ENV_HPP

#include <cctype>
#include <cstdlib>
#include <fftw3.h>
#include <string>

namespace shafft::fft::fftw {

/// Get integer value from environment variable, or default if not set/invalid.
/// @param key Environment variable name
/// @param defv Default value if not set or <= 0
/// @return Parsed value or default
inline int getenvThreadsOrDefault(const char* key, int defv = 1) noexcept {
  if (const char* s = std::getenv(key)) {
    int v = std::atoi(s);
    return v > 0 ? v : defv;
  }
  return defv;
}

/// Parse SHAFFT_FFTW_PLANNER environment variable to FFTW planner flags.
/// Supported values: ESTIMATE (default), MEASURE, PATIENT, EXHAUSTIVE
/// @return FFTW planner flag constant
inline unsigned getenvPlannerFlags() noexcept {
  const char* s = std::getenv("SHAFFT_FFTW_PLANNER");
  if (!s)
    return FFTW_ESTIMATE;
  std::string val(s);
  for (char& c : val)
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  if (val == "MEASURE")
    return FFTW_MEASURE;
  if (val == "PATIENT")
    return FFTW_PATIENT;
  if (val == "EXHAUSTIVE")
    return FFTW_EXHAUSTIVE;
  return FFTW_ESTIMATE;
}

} // namespace shafft::fft::fftw

#endif // SHAFFT_FFT_FFTW_COMMON_ENV_HPP
