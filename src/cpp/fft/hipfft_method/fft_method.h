#ifndef FFT_METHOD_H
#define FFT_METHOD_H

#include "ffthandle.h"
#include <shafft/shafft_types.hpp>
#include <cstdlib>
#include <cstdio>
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

int fftPlan(fftHandle &plan, int nta, int *ta, int ndim, int *dimensions,
            shafft::FFTType precision);

int fftSetStream(fftHandle& plan, hipStream_t stream);

int fftExecute(fftHandle plan, void *data, void *work,
               shafft::FFTDirection direction);

int fftDestroy(fftHandle plan);

#endif // FFT_METHOD_H
