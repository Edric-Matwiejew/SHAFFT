#ifndef KERNEL_H
#define KERNEL_H

#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>

// Normalize kernel with stream support for async execution.
// If stream is nullptr, uses default stream.
// Does NOT synchronize - caller is responsible for synchronization if needed.
int normalizeComplexFloat(float norm_factor, size_t tensor_size, void* data,
                          hipStream_t stream = nullptr);
int normalizeComplexDouble(double norm_factor, size_t tensor_size, void* data,
                           hipStream_t stream = nullptr);

#endif  // KERNEL_H
