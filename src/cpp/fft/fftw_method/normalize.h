#ifndef NORMALIZE_H
#define NORMALIZE_H

#include <cstddef>

/**
 * @brief Scale a tensor of complex numbers in-place (single precision).
 *
 * Each element of @p data (interpreted as fftwf_complex) has its
 * real and imaginary parts multiplied by @p norm_factor.
 *
 * @param norm_factor Scaling factor.
 * @param tensor_size Number of complex elements to scale.
 * @param data Pointer to fftwf_complex array.
 * @return 0 on success (FFTW normalize always succeeds).
 */
int normalizeComplexFloat(float norm_factor,
                          std::size_t tensor_size,
                          void* data);

/**
 * @brief Scale a tensor of complex numbers in-place (double precision).
 *
 * Each element of @p data (interpreted as fftw_complex) has its
 * real and imaginary parts multiplied by @p norm_factor.
 *
 * @param norm_factor Scaling factor.
 * @param tensor_size Number of complex elements to scale.
 * @param data Pointer to fftw_complex array.
 * @return 0 on success (FFTW normalize always succeeds).
 */
int normalizeComplexDouble(double norm_factor,
                           std::size_t tensor_size,
                           void* data);

#endif // NORMALIZE_H
