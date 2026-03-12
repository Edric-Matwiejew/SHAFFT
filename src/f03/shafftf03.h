#ifndef SHAFFT_F03_H
#define SHAFFT_F03_H

#include <shafft/shafft.h>

#include <mpi.h>

extern "C" {

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
                             MPI_Fint* f_handle);

int shafftNDInitf03(void* planPtr,
                    int ndim,
                    int commDims[],
                    int dimensions[],
                    shafft_t precision,
                    MPI_Fint* f_handle,
                    shafft_transform_layout_t* output_policy);

// 1D FFT wrappers
int shafftConfiguration1Df03(size_t N,
                             size_t* localN,
                             size_t* localStart,
                             size_t* localAllocSize,
                             shafft_t precision,
                             MPI_Fint* f_handle);

int shafft1DInitf03(void* planPtr,
                    size_t N,
                    size_t localN,
                    size_t localStart,
                    shafft_t precision,
                    MPI_Fint* f_handle);

// Config struct alloc/free helpers (Fortran needs heap-allocated C struct)
shafft_nd_config_t* shafftConfigNDAllocf03(void);
void shafftConfigNDFreef03(shafft_nd_config_t* cfg);
shafft_1d_config_t* shafftConfig1DAllocf03(void);
void shafftConfig1DFreef03(shafft_1d_config_t* cfg);

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
                          MPI_Fint* fComm);
int shafftConfigNDResolvef03(shafft_nd_config_t* cfg);
int shafftNDInitFromConfigf03(void* planPtr, shafft_nd_config_t* cfg);
int shafftConfig1DInitf03(shafft_1d_config_t* cfg,
                          size_t globalSize,
                          shafft_t precision,
                          MPI_Fint* fComm);
int shafftConfig1DResolvef03(shafft_1d_config_t* cfg);
int shafft1DInitFromConfigf03(void* planPtr, shafft_1d_config_t* cfg);
int shafftGetCommunicatorf03(void* planPtr, MPI_Fint* outFint);

// ---- C accessor functions for Fortran sync procedures ----
// These are NOT part of the public C API; they back the Fortran derived-type sync.

// ND scalar getters
int shafftConfigNDGetNdim(const shafft_nd_config_t* cfg);
int shafftConfigNDGetPrecision(const shafft_nd_config_t* cfg);
size_t shafftConfigNDGetAllocElements(const shafft_nd_config_t* cfg);
int shafftConfigNDGetIsActive(const shafft_nd_config_t* cfg);
int shafftConfigNDGetActiveRank(const shafft_nd_config_t* cfg);
int shafftConfigNDGetActiveSize(const shafft_nd_config_t* cfg);
int shafftConfigNDGetNda(const shafft_nd_config_t* cfg);
int shafftConfigNDGetCommSize(const shafft_nd_config_t* cfg);
int shafftConfigNDGetStatus(const shafft_nd_config_t* cfg);
int shafftConfigNDGetOutputPolicy(const shafft_nd_config_t* cfg);
int shafftConfigNDGetStrategy(const shafft_nd_config_t* cfg);
size_t shafftConfigNDGetMemLimit(const shafft_nd_config_t* cfg);
int shafftConfigNDGetHintNda(const shafft_nd_config_t* cfg);
int shafftConfigNDGetFlags(const shafft_nd_config_t* cfg);
MPI_Fint shafftConfigNDGetWorldComm(const shafft_nd_config_t* cfg);
MPI_Fint shafftConfigNDGetActiveComm(const shafft_nd_config_t* cfg);

// ND topology getters
int shafftConfigNDGetNodeId(const shafft_nd_config_t* cfg);
int shafftConfigNDGetNodeCount(const shafft_nd_config_t* cfg);
size_t shafftConfigNDGetHostnameLen(const shafft_nd_config_t* cfg);
void shafftConfigNDGetHostname(const shafft_nd_config_t* cfg, char* out, size_t maxLen);
size_t shafftConfigNDGetDeviceNameLen(const shafft_nd_config_t* cfg);
void shafftConfigNDGetDeviceName(const shafft_nd_config_t* cfg, char* out, size_t maxLen);

// ND array getters (copy into caller's buffer)
void shafftConfigNDGetGlobalShape(const shafft_nd_config_t* cfg, size_t* out);
void shafftConfigNDGetCommDims(const shafft_nd_config_t* cfg, int* out);
void shafftConfigNDGetHintCommDims(const shafft_nd_config_t* cfg, int* out);
void shafftConfigNDGetInitialSubsize(const shafft_nd_config_t* cfg, size_t* out);
void shafftConfigNDGetInitialOffset(const shafft_nd_config_t* cfg, size_t* out);
void shafftConfigNDGetOutputSubsize(const shafft_nd_config_t* cfg, size_t* out);
void shafftConfigNDGetOutputOffset(const shafft_nd_config_t* cfg, size_t* out);

// ND setters (for modifiable fields — used by sync_to_c before re-resolve)
void shafftConfigNDSetOutputPolicy(shafft_nd_config_t* cfg, int pol);
void shafftConfigNDSetStrategy(shafft_nd_config_t* cfg, int s);
void shafftConfigNDSetMemLimit(shafft_nd_config_t* cfg, size_t limit);
void shafftConfigNDSetHintNda(shafft_nd_config_t* cfg, int nda);
void shafftConfigNDSetHintCommDims(shafft_nd_config_t* cfg, const int* dims);

// 1D scalar getters
size_t shafftConfig1DGetGlobalSize(const shafft_1d_config_t* cfg);
int shafftConfig1DGetPrecision(const shafft_1d_config_t* cfg);
size_t shafftConfig1DGetAllocElements(const shafft_1d_config_t* cfg);
int shafftConfig1DGetIsActive(const shafft_1d_config_t* cfg);
int shafftConfig1DGetActiveRank(const shafft_1d_config_t* cfg);
int shafftConfig1DGetActiveSize(const shafft_1d_config_t* cfg);
int shafftConfig1DGetStatus(const shafft_1d_config_t* cfg);
int shafftConfig1DGetFlags(const shafft_1d_config_t* cfg);
MPI_Fint shafftConfig1DGetWorldComm(const shafft_1d_config_t* cfg);
MPI_Fint shafftConfig1DGetActiveComm(const shafft_1d_config_t* cfg);

// 1D topology getters
int shafftConfig1DGetNodeId(const shafft_1d_config_t* cfg);
int shafftConfig1DGetNodeCount(const shafft_1d_config_t* cfg);
size_t shafftConfig1DGetHostnameLen(const shafft_1d_config_t* cfg);
void shafftConfig1DGetHostname(const shafft_1d_config_t* cfg, char* out, size_t maxLen);
size_t shafftConfig1DGetDeviceNameLen(const shafft_1d_config_t* cfg);
void shafftConfig1DGetDeviceName(const shafft_1d_config_t* cfg, char* out, size_t maxLen);

// 1D layout getters
size_t shafftConfig1DGetInitialLocalSize(const shafft_1d_config_t* cfg);
size_t shafftConfig1DGetInitialLocalStart(const shafft_1d_config_t* cfg);
size_t shafftConfig1DGetOutputLocalSize(const shafft_1d_config_t* cfg);
size_t shafftConfig1DGetOutputLocalStart(const shafft_1d_config_t* cfg);
}

#endif // SHAFFT_F03_H
