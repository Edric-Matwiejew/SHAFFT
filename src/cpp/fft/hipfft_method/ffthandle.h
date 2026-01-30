#ifndef FFT_HANDLE_H
#define FFT_HANDLE_H
#include <hipfft/hipfft.h>
struct fftHandle {
  hipfftHandle *subplans = nullptr;
  int nsubplans = 0;
  int *superbatches = nullptr;
  int *superbatches_offset = nullptr;
  hipfftType_t fft_type;
  hipStream_t stream = nullptr;
};
#endif // FFT_HANDLE_H
