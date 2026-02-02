// shafft_error.cpp
#include <shafft/shafft_error.hpp>

#include <cstdio>
#include <cstring>

// Thread-local slots (per thread)
static thread_local int g_status = static_cast<int>(shafft::Status::SHAFFT_SUCCESS);
static thread_local int g_source = SHAFFT_ERRSRC_NONE;
static thread_local int g_code = 0;

// HIP/hipFFT error string helpers (only available when HIP backend is used)
#if SHAFFT_BACKEND_HIPFFT
#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

static const char* hipfft_error_string(hipfftResult_t result) {
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

int shafft_last_error_status(void) {
  return g_status;
}
int shafft_last_error_source(void) {
  return g_source;
}
int shafft_last_error_domain_code(void) {
  return g_code;
}

void shafft_clear_last_error(void) {
  g_status = static_cast<int>(shafft::Status::SHAFFT_SUCCESS);
  g_source = SHAFFT_ERRSRC_NONE;
  g_code = 0;
}

const char* shafft_error_source_name(int source) {
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

int shafft_last_error_message(char* buf, int buflen) {
  if (!buf || buflen <= 0)
    return 0;
  buf[0] = '\0';

  // MPI domain: ask MPI for a human-readable string
  if (g_source == SHAFFT_ERRSRC_MPI && g_code != MPI_SUCCESS) {
    int n = 0;
    MPI_Error_string(g_code, buf, &n);
    if (n >= buflen)
      n = buflen - 1;
    buf[n] = '\0';
    return n;
  }

#if SHAFFT_BACKEND_HIPFFT
  // HIP runtime errors
  if (g_source == SHAFFT_ERRSRC_HIP) {
    const char* msg = hipGetErrorString(static_cast<hipError_t>(g_code));
    int n = snprintf(buf, buflen, "HIP error %d: %s", g_code, msg ? msg : "unknown");
    if (n >= buflen)
      n = buflen - 1;
    buf[n] = '\0';
    return n;
  }

  // hipFFT errors
  if (g_source == SHAFFT_ERRSRC_HIPFFT) {
    const char* msg = hipfft_error_string(static_cast<hipfftResult_t>(g_code));
    int n = snprintf(buf, buflen, "hipFFT error %d: %s", g_code, msg);
    if (n >= buflen)
      n = buflen - 1;
    buf[n] = '\0';
    return n;
  }
#endif

  // FFTW errors (FFTW doesn't have error codes, so just a generic message)
  if (g_source == SHAFFT_ERRSRC_FFTW) {
    int n = snprintf(buf, buflen, "FFTW error: plan creation or execution failed");
    if (n >= buflen)
      n = buflen - 1;
    buf[n] = '\0';
    return n;
  }

  // System errors
  if (g_source == SHAFFT_ERRSRC_SYSTEM) {
    int n = snprintf(buf, buflen, "System error %d", g_code);
    if (n >= buflen)
      n = buflen - 1;
    buf[n] = '\0';
    return n;
  }

  // None/unknown: no message
  return 0;
}

}  // extern "C"

namespace shafft {
namespace detail {

void set_last_error(Status st, shafft_errsrc_t src, int domain_code) noexcept {
  g_status = static_cast<int>(st);
  g_source = static_cast<int>(src);
  g_code = domain_code;
}

}  // namespace detail
}  // namespace shafft
