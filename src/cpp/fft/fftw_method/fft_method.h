#ifndef FFT_METHOD_H
#define FFT_METHOD_H

#include "ffthandle.h"
#include <shafft/shafft_types.hpp>

/**
 * @brief Plan a contiguous multi-dimensional FFT using FFTW (guru64 interface).
 *
 * @param plan       Reference to an fftHandle to be initialised.
 * @param nta        Number of contiguous transform axes.
 * @param ta         Array of contiguous transform axes.
 * @param ndim       Total tensor rank.
 * @param dimensions Array of tensor dimensions (length = ndim).
 * @param precision  Precision type (C2C for single, Z2Z for double).
 * @return           0 on success or throws std::exception on error.
 */
int fftPlan(fftHandle &plan,
            int nta, int *ta,
            int ndim, int *dimensions,
            shafft::FFTType precision);

/**
 * @brief Execute an FFT transform.
 *
 * @param plan       FFT plan handle created by fftPlan.
 * @param data       Input buffer pointer.
 * @param work       Output buffer pointer (receives result).
 * @param direction  Transform direction (FORWARD or BACKWARD).
 * @return           0 on success.
 */
int fftExecute(fftHandle plan,
               void *data, void *work,
               shafft::FFTDirection direction);

/**
 * @brief Destroy an FFT plan and free all associated FFTW resources.
 *
 * @param plan FFT plan handle.
 * @return     0 on success.
 */
int fftDestroy(fftHandle plan);

#endif // FFT_METHOD_H
