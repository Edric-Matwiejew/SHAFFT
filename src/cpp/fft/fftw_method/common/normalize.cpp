#include "normalize.hpp"

#include <cstddef>
#include <fftw3.h>

int normalizeComplexFloat(float normFactor, std::size_t tensorSize, void* data) {
  if (!data || tensorSize == 0)
    return 0;

  auto* arr = reinterpret_cast<fftwf_complex*>(data);

#pragma omp parallel for schedule(static)
  for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(tensorSize); ++i) {
    arr[i][0] *= normFactor;
    arr[i][1] *= normFactor;
  }
  return 0;
}

int normalizeComplexDouble(double normFactor, std::size_t tensorSize, void* data) {
  if (!data || tensorSize == 0)
    return 0;

  auto* arr = reinterpret_cast<fftw_complex*>(data);

#pragma omp parallel for schedule(static)
  for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(tensorSize); ++i) {
    arr[i][0] *= normFactor;
    arr[i][1] *= normFactor;
  }
  return 0;
}
