#ifndef SHAFFT_NORMALIZE_HPP
#define SHAFFT_NORMALIZE_HPP

#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>

/// Normalize complex float array by scalar factor (asynchronous, no sync).
int normalizeComplexFloat(float norm_factor,
                          size_t tensor_size,
                          void* data,
                          hipStream_t stream = nullptr);

/// Normalize complex double array by scalar factor (asynchronous, no sync).
int normalizeComplexDouble(double norm_factor,
                           size_t tensor_size,
                           void* data,
                           hipStream_t stream = nullptr);

#endif // SHAFFT_NORMALIZE_HPP
