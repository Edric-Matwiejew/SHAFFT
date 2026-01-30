#ifndef KERNEL_H
#define KERNEL_H

#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>

int normalizeComplexFloat(float norm_factor, size_t tensor_size, void *data);
int normalizeComplexDouble(double norm_factor, size_t tensor_size, void *data);

#endif // KERNEL_H
