#ifndef SHAFFT_HIPCHECK_HPP
#define SHAFFT_HIPCHECK_HPP

#include <cstdio>

/// Check HIP runtime call and print error with location on failure.
/// Returns 0 on success, HIP error code on failure.
#define hipCheck(stmt)                                                                             \
  ([&]() -> int {                                                                                  \
    hipError_t err = (stmt);                                                                       \
    if (err != hipSuccess) {                                                                       \
      fprintf(stderr,                                                                              \
              "HIP error: %s\n%s (%d) at %s:%d\n",                                                 \
              #stmt,                                                                               \
              hipGetErrorString(err),                                                              \
              static_cast<int>(err),                                                               \
              __FILE__,                                                                            \
              __LINE__);                                                                           \
      return static_cast<int>(err);                                                                \
    }                                                                                              \
    return 0;                                                                                      \
  })()

/// Check hipFFT call and print error with location on failure.
/// Returns 0 on success, hipFFT result code on failure.
#define hipfftCheck(stmt)                                                                          \
  ([&]() -> int {                                                                                  \
    hipfftResult_t res = (stmt);                                                                   \
    if (res != HIPFFT_SUCCESS) {                                                                   \
      fprintf(stderr,                                                                              \
              "hipFFT error: %s returned %d at %s:%d\n",                                           \
              #stmt,                                                                               \
              static_cast<int>(res),                                                               \
              __FILE__,                                                                            \
              __LINE__);                                                                           \
      return static_cast<int>(res);                                                                \
    }                                                                                              \
    return 0;                                                                                      \
  })()

#endif // SHAFFT_HIPCHECK_HPP
