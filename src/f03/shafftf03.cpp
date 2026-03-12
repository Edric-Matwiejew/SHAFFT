#include "shafftf03.h"

#include <shafft/shafft.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>

// N-D FFT wrappers (convert Fortran MPI handle to C communicator)

int shafftConfigurationNDf03(int ndim,
                             int* size,
                             shafft_t precision,
                             int* commDims,
                             int* nda,
                             size_t* subsize,
                             size_t* offset,
                             int* commSize,
                             shafft_decomposition_strategy_t strategy,
                             size_t memLimit,
                             MPI_Fint* fHandle) {
  MPI_Comm cComm = MPI_Comm_f2c(*fHandle);
  return shafftConfigurationND(
      ndim, size, precision, commDims, nda, subsize, offset, commSize, strategy, memLimit, cComm);
}

int shafftNDInitf03(void* planPtr,
                    int ndim,
                    int commDims[],
                    int dimensions[],
                    shafft_t precision,
                    MPI_Fint* fHandle,
                    shafft_transform_layout_t* output_policy) {
  MPI_Comm cComm = MPI_Comm_f2c(*fHandle);
  if (!output_policy) {
    return static_cast<int>(SHAFFT_ERR_NULLPTR);
  }
  return shafftNDInit(planPtr, ndim, commDims, dimensions, precision, cComm, *output_policy);
}

// 1D FFT wrappers (convert Fortran MPI handle to C communicator)

int shafftConfiguration1Df03(size_t globalN,
                             size_t* localN,
                             size_t* localStart,
                             size_t* localAllocSize,
                             shafft_t precision,
                             MPI_Fint* fHandle) {
  MPI_Comm cComm = MPI_Comm_f2c(*fHandle);
  return shafftConfiguration1D(globalN, localN, localStart, localAllocSize, precision, cComm);
}

int shafft1DInitf03(void* planPtr,
                    size_t globalN,
                    size_t localN,
                    size_t localStart,
                    shafft_t precision,
                    MPI_Fint* fHandle) {
  MPI_Comm cComm = MPI_Comm_f2c(*fHandle);
  return shafft1DInit(planPtr, globalN, localN, localStart, precision, cComm);
}

// Config struct alloc/free helpers (Fortran needs heap-allocated C struct)

shafft_nd_config_t* shafftConfigNDAllocf03(void) {
  return static_cast<shafft_nd_config_t*>(std::calloc(1, sizeof(shafft_nd_config_t)));
}

void shafftConfigNDFreef03(shafft_nd_config_t* cfg) {
  std::free(cfg);
}

shafft_1d_config_t* shafftConfig1DAllocf03(void) {
  return static_cast<shafft_1d_config_t*>(std::calloc(1, sizeof(shafft_1d_config_t)));
}

void shafftConfig1DFreef03(shafft_1d_config_t* cfg) {
  std::free(cfg);
}

// Config object wrappers (convert Fortran MPI handle to C communicator)

int shafftConfigNDInitf03(shafft_nd_config_t* cfg,
                          int ndim,
                          const size_t* globalShape,
                          shafft_t precision,
                          const int* commDims,
                          int hintNda,
                          shafft_decomposition_strategy_t strategy,
                          shafft_transform_layout_t outputPolicy,
                          size_t memLimit,
                          MPI_Fint* fComm) {
  MPI_Comm cComm = MPI_Comm_f2c(*fComm);
  return shafftConfigNDInit(cfg,
                            ndim,
                            globalShape,
                            precision,
                            commDims,
                            hintNda,
                            strategy,
                            outputPolicy,
                            memLimit,
                            cComm);
}

int shafftConfigNDResolvef03(shafft_nd_config_t* cfg) {
  return shafftConfigNDResolve(cfg);
}

int shafftNDInitFromConfigf03(void* planPtr, shafft_nd_config_t* cfg) {
  return shafftNDInitFromConfig(planPtr, cfg);
}

int shafftConfig1DInitf03(shafft_1d_config_t* cfg,
                          size_t globalSize,
                          shafft_t precision,
                          MPI_Fint* fComm) {
  MPI_Comm cComm = MPI_Comm_f2c(*fComm);
  return shafftConfig1DInit(cfg, globalSize, precision, cComm);
}

int shafftConfig1DResolvef03(shafft_1d_config_t* cfg) {
  return shafftConfig1DResolve(cfg);
}

int shafft1DInitFromConfigf03(void* planPtr, shafft_1d_config_t* cfg) {
  return shafft1DInitFromConfig(planPtr, cfg);
}

int shafftGetCommunicatorf03(void* planPtr, MPI_Fint* outFint) {
  MPI_Comm cComm = MPI_COMM_NULL;
  int rc = shafftGetCommunicator(planPtr, &cComm);
  if (rc != 0)
    return rc;
  *outFint = MPI_Comm_c2f(cComm);
  return 0;
}

// ---- C accessor functions for Fortran sync procedures ----

// Helper to copy a string to output buffer
static void copyStr(const char* src, size_t srcLen, char* out, size_t maxLen) {
  if (!out || maxLen == 0)
    return;
  size_t n = std::min(srcLen, maxLen);
  if (src && n > 0)
    std::memcpy(out, src, n);
  if (n < maxLen)
    out[n] = '\0';
}

// ---- ND scalar getters ----
int shafftConfigNDGetNdim(const shafft_nd_config_t* cfg) {
  return cfg ? cfg->ndim : 0;
}
int shafftConfigNDGetPrecision(const shafft_nd_config_t* cfg) {
  return cfg ? static_cast<int>(cfg->precision) : 0;
}
size_t shafftConfigNDGetAllocElements(const shafft_nd_config_t* cfg) {
  return cfg ? cfg->allocElements : 0;
}
int shafftConfigNDGetIsActive(const shafft_nd_config_t* cfg) {
  return cfg ? cfg->isActive : 0;
}
int shafftConfigNDGetActiveRank(const shafft_nd_config_t* cfg) {
  return cfg ? cfg->activeRank : -1;
}
int shafftConfigNDGetActiveSize(const shafft_nd_config_t* cfg) {
  return cfg ? cfg->activeSize : 0;
}
int shafftConfigNDGetNda(const shafft_nd_config_t* cfg) {
  return cfg ? cfg->nda : 0;
}
int shafftConfigNDGetCommSize(const shafft_nd_config_t* cfg) {
  return cfg ? cfg->commSize : 0;
}
int shafftConfigNDGetStatus(const shafft_nd_config_t* cfg) {
  return cfg ? cfg->status : 0;
}
int shafftConfigNDGetOutputPolicy(const shafft_nd_config_t* cfg) {
  return cfg ? static_cast<int>(cfg->outputPolicy) : 0;
}
int shafftConfigNDGetStrategy(const shafft_nd_config_t* cfg) {
  return cfg ? static_cast<int>(cfg->strategy) : 0;
}
size_t shafftConfigNDGetMemLimit(const shafft_nd_config_t* cfg) {
  return cfg ? cfg->memLimit : 0;
}
int shafftConfigNDGetHintNda(const shafft_nd_config_t* cfg) {
  return cfg ? cfg->hintNda : 0;
}
int shafftConfigNDGetFlags(const shafft_nd_config_t* cfg) {
  return cfg ? cfg->flags : 0;
}

MPI_Fint shafftConfigNDGetWorldComm(const shafft_nd_config_t* cfg) {
  return cfg ? MPI_Comm_c2f(cfg->worldComm) : MPI_Comm_c2f(MPI_COMM_NULL);
}
MPI_Fint shafftConfigNDGetActiveComm(const shafft_nd_config_t* cfg) {
  return cfg ? MPI_Comm_c2f(cfg->activeComm) : MPI_Comm_c2f(MPI_COMM_NULL);
}

// ND topology getters
int shafftConfigNDGetNodeId(const shafft_nd_config_t* cfg) {
  return cfg ? cfg->nodeId : 0;
}
int shafftConfigNDGetNodeCount(const shafft_nd_config_t* cfg) {
  return cfg ? cfg->nodeCount : 0;
}
size_t shafftConfigNDGetHostnameLen(const shafft_nd_config_t* cfg) {
  return cfg ? cfg->hostnameLen : 0;
}
void shafftConfigNDGetHostname(const shafft_nd_config_t* cfg, char* out, size_t maxLen) {
  if (cfg)
    copyStr(cfg->hostname, cfg->hostnameLen, out, maxLen);
}
size_t shafftConfigNDGetDeviceNameLen(const shafft_nd_config_t* cfg) {
  return cfg ? cfg->deviceNameLen : 0;
}
void shafftConfigNDGetDeviceName(const shafft_nd_config_t* cfg, char* out, size_t maxLen) {
  if (cfg)
    copyStr(cfg->deviceName, cfg->deviceNameLen, out, maxLen);
}

// ND array getters
void shafftConfigNDGetGlobalShape(const shafft_nd_config_t* cfg, size_t* out) {
  if (cfg && out && cfg->globalShape)
    std::memcpy(out, cfg->globalShape, static_cast<size_t>(cfg->ndim) * sizeof(size_t));
}
void shafftConfigNDGetCommDims(const shafft_nd_config_t* cfg, int* out) {
  if (cfg && out && cfg->commDims)
    std::memcpy(out, cfg->commDims, static_cast<size_t>(cfg->ndim) * sizeof(int));
}
void shafftConfigNDGetHintCommDims(const shafft_nd_config_t* cfg, int* out) {
  if (cfg && out && cfg->hintCommDims)
    std::memcpy(out, cfg->hintCommDims, static_cast<size_t>(cfg->ndim) * sizeof(int));
}
void shafftConfigNDGetInitialSubsize(const shafft_nd_config_t* cfg, size_t* out) {
  if (cfg && out && cfg->initial.subsize)
    std::memcpy(out, cfg->initial.subsize, static_cast<size_t>(cfg->ndim) * sizeof(size_t));
}
void shafftConfigNDGetInitialOffset(const shafft_nd_config_t* cfg, size_t* out) {
  if (cfg && out && cfg->initial.offset)
    std::memcpy(out, cfg->initial.offset, static_cast<size_t>(cfg->ndim) * sizeof(size_t));
}
void shafftConfigNDGetOutputSubsize(const shafft_nd_config_t* cfg, size_t* out) {
  if (cfg && out && cfg->output.subsize)
    std::memcpy(out, cfg->output.subsize, static_cast<size_t>(cfg->ndim) * sizeof(size_t));
}
void shafftConfigNDGetOutputOffset(const shafft_nd_config_t* cfg, size_t* out) {
  if (cfg && out && cfg->output.offset)
    std::memcpy(out, cfg->output.offset, static_cast<size_t>(cfg->ndim) * sizeof(size_t));
}

// ND setters
void shafftConfigNDSetOutputPolicy(shafft_nd_config_t* cfg, int pol) {
  if (cfg)
    cfg->outputPolicy = static_cast<shafft_transform_layout_t>(pol);
}
void shafftConfigNDSetStrategy(shafft_nd_config_t* cfg, int s) {
  if (cfg)
    cfg->strategy = static_cast<shafft_decomposition_strategy_t>(s);
}
void shafftConfigNDSetMemLimit(shafft_nd_config_t* cfg, size_t limit) {
  if (cfg)
    cfg->memLimit = limit;
}
void shafftConfigNDSetHintNda(shafft_nd_config_t* cfg, int nda) {
  if (cfg)
    cfg->hintNda = nda;
}
void shafftConfigNDSetHintCommDims(shafft_nd_config_t* cfg, const int* dims) {
  if (cfg && dims && cfg->hintCommDims)
    std::memcpy(cfg->hintCommDims, dims, static_cast<size_t>(cfg->ndim) * sizeof(int));
}

// ---- 1D scalar getters ----
size_t shafftConfig1DGetGlobalSize(const shafft_1d_config_t* cfg) {
  return cfg ? cfg->globalSize : 0;
}
int shafftConfig1DGetPrecision(const shafft_1d_config_t* cfg) {
  return cfg ? static_cast<int>(cfg->precision) : 0;
}
size_t shafftConfig1DGetAllocElements(const shafft_1d_config_t* cfg) {
  return cfg ? cfg->allocElements : 0;
}
int shafftConfig1DGetIsActive(const shafft_1d_config_t* cfg) {
  return cfg ? cfg->isActive : 0;
}
int shafftConfig1DGetActiveRank(const shafft_1d_config_t* cfg) {
  return cfg ? cfg->activeRank : -1;
}
int shafftConfig1DGetActiveSize(const shafft_1d_config_t* cfg) {
  return cfg ? cfg->activeSize : 0;
}
int shafftConfig1DGetStatus(const shafft_1d_config_t* cfg) {
  return cfg ? cfg->status : 0;
}
int shafftConfig1DGetFlags(const shafft_1d_config_t* cfg) {
  return cfg ? cfg->flags : 0;
}

MPI_Fint shafftConfig1DGetWorldComm(const shafft_1d_config_t* cfg) {
  return cfg ? MPI_Comm_c2f(cfg->worldComm) : MPI_Comm_c2f(MPI_COMM_NULL);
}
MPI_Fint shafftConfig1DGetActiveComm(const shafft_1d_config_t* cfg) {
  return cfg ? MPI_Comm_c2f(cfg->activeComm) : MPI_Comm_c2f(MPI_COMM_NULL);
}

// 1D topology getters
int shafftConfig1DGetNodeId(const shafft_1d_config_t* cfg) {
  return cfg ? cfg->nodeId : 0;
}
int shafftConfig1DGetNodeCount(const shafft_1d_config_t* cfg) {
  return cfg ? cfg->nodeCount : 0;
}
size_t shafftConfig1DGetHostnameLen(const shafft_1d_config_t* cfg) {
  return cfg ? cfg->hostnameLen : 0;
}
void shafftConfig1DGetHostname(const shafft_1d_config_t* cfg, char* out, size_t maxLen) {
  if (cfg)
    copyStr(cfg->hostname, cfg->hostnameLen, out, maxLen);
}
size_t shafftConfig1DGetDeviceNameLen(const shafft_1d_config_t* cfg) {
  return cfg ? cfg->deviceNameLen : 0;
}
void shafftConfig1DGetDeviceName(const shafft_1d_config_t* cfg, char* out, size_t maxLen) {
  if (cfg)
    copyStr(cfg->deviceName, cfg->deviceNameLen, out, maxLen);
}

// 1D layout getters
size_t shafftConfig1DGetInitialLocalSize(const shafft_1d_config_t* cfg) {
  return cfg ? cfg->initial.localSize : 0;
}
size_t shafftConfig1DGetInitialLocalStart(const shafft_1d_config_t* cfg) {
  return cfg ? cfg->initial.localStart : 0;
}
size_t shafftConfig1DGetOutputLocalSize(const shafft_1d_config_t* cfg) {
  return cfg ? cfg->output.localSize : 0;
}
size_t shafftConfig1DGetOutputLocalStart(const shafft_1d_config_t* cfg) {
  return cfg ? cfg->output.localStart : 0;
}
