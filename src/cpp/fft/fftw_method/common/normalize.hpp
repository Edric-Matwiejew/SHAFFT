#ifndef SHAFFT_FFTW_NORMALIZE_HPP
#define SHAFFT_FFTW_NORMALIZE_HPP

#include <cstddef>

int normalizeComplexFloat(float norm_factor, std::size_t tensor_size, void* data);

int normalizeComplexDouble(double norm_factor, std::size_t tensor_size, void* data);

#endif // SHAFFT_FFTW_NORMALIZE_HPP
