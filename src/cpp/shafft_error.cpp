#include <shafft/shafft.h> // for shafft_errsrc_t and related enums
#include <shafft/shafft_error.hpp>
#include <shafft/shafft_types.hpp>

#include <cstdio>
#include <cstring>

// Thread-local slots (per thread)
static thread_local int gStatus = static_cast<int>(shafft::Status::SUCCESS);
static thread_local int gSource = SHAFFT_ERRSRC_NONE;
static thread_local int gCode = 0;

// HIP/hipFFT error string helpers (only available when HIP backend is used)
#if SHAFFT_BACKEND_HIPFFT
#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

static const char* hipfftErrorString(hipfftResult_t result) {
  switch (result) {
  case HIPFFT_SUCCESS:
    return "HIPFFT_SUCCESS";
  case HIPFFT_INVALID_PLAN:
    return "HIPFFT_INVALID_PLAN";
  case HIPFFT_ALLOC_FAILED:
    return "HIPFFT_ALLOC_FAILED";
  case HIPFFT_INVALID_TYPE:
    return "HIPFFT_INVALID_TYPE";
  case HIPFFT_INVALID_VALUE:
    return "HIPFFT_INVALID_VALUE";
  case HIPFFT_INTERNAL_ERROR:
    return "HIPFFT_INTERNAL_ERROR";
  case HIPFFT_EXEC_FAILED:
    return "HIPFFT_EXEC_FAILED";
  case HIPFFT_SETUP_FAILED:
    return "HIPFFT_SETUP_FAILED";
  case HIPFFT_INVALID_SIZE:
    return "HIPFFT_INVALID_SIZE";
  case HIPFFT_UNALIGNED_DATA:
    return "HIPFFT_UNALIGNED_DATA";
  case HIPFFT_INCOMPLETE_PARAMETER_LIST:
    return "HIPFFT_INCOMPLETE_PARAMETER_LIST";
  case HIPFFT_INVALID_DEVICE:
    return "HIPFFT_INVALID_DEVICE";
  case HIPFFT_PARSE_ERROR:
    return "HIPFFT_PARSE_ERROR";
  case HIPFFT_NO_WORKSPACE:
    return "HIPFFT_NO_WORKSPACE";
  case HIPFFT_NOT_IMPLEMENTED:
    return "HIPFFT_NOT_IMPLEMENTED";
  case HIPFFT_NOT_SUPPORTED:
    return "HIPFFT_NOT_SUPPORTED";
  default:
    return "HIPFFT_UNKNOWN_ERROR";
  }
}
#endif

extern "C" {

int shafftLastErrorStatus(void) {
  return gStatus;
}
int shafftLastErrorSource(void) {
  return gSource;
}
int shafftLastErrorDomainCode(void) {
  return gCode;
}

void shafftClearLastError(void) {
  gStatus = static_cast<int>(shafft::Status::SUCCESS);
  gSource = SHAFFT_ERRSRC_NONE;
  gCode = 0;
}

const char* shafftErrorSourceName(int source) {
  switch (static_cast<shafft_errsrc_t>(source)) {
  case SHAFFT_ERRSRC_NONE:
    return "SHAFFT";
  case SHAFFT_ERRSRC_MPI:
    return "MPI";
  case SHAFFT_ERRSRC_HIP:
    return "HIP";
  case SHAFFT_ERRSRC_HIPFFT:
    return "hipFFT";
  case SHAFFT_ERRSRC_FFTW:
    return "FFTW";
  case SHAFFT_ERRSRC_SYSTEM:
    return "System";
  default:
    return "Unknown";
  }
}

int shafftLastErrorMessage(char* buf, int buflen) {
  if (!buf || buflen <= 0)
    return 0;
  buf[0] = '\0';

  // MPI domain: ask MPI for a human-readable string
  if (gSource == SHAFFT_ERRSRC_MPI && gCode != MPI_SUCCESS) {
    int n = 0;
    MPI_Error_string(gCode, buf, &n);
    if (n >= buflen)
      n = buflen - 1;
    buf[n] = '\0';
    return n;
  }

#if SHAFFT_BACKEND_HIPFFT
  // HIP runtime errors
  if (gSource == SHAFFT_ERRSRC_HIP) {
    const char* msg = hipGetErrorString(static_cast<hipError_t>(gCode));
    int n = snprintf(buf, buflen, "HIP error %d: %s", gCode, msg ? msg : "unknown");
    if (n >= buflen)
      n = buflen - 1;
    buf[n] = '\0';
    return n;
  }

  // hipFFT errors
  if (gSource == SHAFFT_ERRSRC_HIPFFT) {
    const char* msg = hipfftErrorString(static_cast<hipfftResult_t>(gCode));
    int n = snprintf(buf, buflen, "hipFFT error %d: %s", gCode, msg);
    if (n >= buflen)
      n = buflen - 1;
    buf[n] = '\0';
    return n;
  }
#endif

  // FFTW errors (FFTW doesn't have error codes, so just a generic message)
  if (gSource == SHAFFT_ERRSRC_FFTW) {
    int n = snprintf(buf, buflen, "FFTW error: plan creation or execution failed");
    if (n >= buflen)
      n = buflen - 1;
    buf[n] = '\0';
    return n;
  }

  // System errors
  if (gSource == SHAFFT_ERRSRC_SYSTEM) {
    int n = snprintf(buf, buflen, "System error %d", gCode);
    if (n >= buflen)
      n = buflen - 1;
    buf[n] = '\0';
    return n;
  }

  // None/unknown: no message
  return 0;
}

} // extern "C"

namespace shafft {
namespace detail {

void setLastError(Status st, shafft_errsrc_t src, int domainCode) noexcept {
  gStatus = static_cast<int>(st);
  gSource = static_cast<int>(src);
  gCode = domainCode;
}

} // namespace detail
} // namespace shafft
