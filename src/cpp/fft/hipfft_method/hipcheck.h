#ifndef HIPCHECK_H
#define HIPCHECK_H

#include <cstdio>

// hipCheck - Checks HIP runtime calls and returns error code on failure
// Returns 0 on success, non-zero HIP error code on failure
#define hipCheck(stmt)                                                         \
  ([&]() -> int {                                                              \
    hipError_t err = (stmt);                                                   \
    if (err != hipSuccess) {                                                   \
      fprintf(stderr, "HIP error: %s\n%s (%d) at %s:%d\n", #stmt,              \
              hipGetErrorString(err), static_cast<int>(err),                   \
              __FILE__, __LINE__);                                             \
      return static_cast<int>(err);                                            \
    }                                                                          \
    return 0;                                                                  \
  })()

// hipfftCheck - Checks hipFFT calls and returns error code on failure
// Returns 0 on success, non-zero hipFFT error code on failure
#define hipfftCheck(stmt)                                                      \
  ([&]() -> int {                                                              \
    hipfftResult_t res = (stmt);                                               \
    if (res != HIPFFT_SUCCESS) {                                               \
      fprintf(stderr, "hipFFT error: %s returned %d at %s:%d\n", #stmt,        \
              static_cast<int>(res), __FILE__, __LINE__);                      \
      return static_cast<int>(res);                                            \
    }                                                                          \
    return 0;                                                                  \
  })()

#endif // HIPCHECK_H
