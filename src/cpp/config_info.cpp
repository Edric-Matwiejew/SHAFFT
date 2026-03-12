/// @file config_info.cpp
/// @brief Implementation of config object lifecycle, resolution, and topology.

#include "detail/config_info.hpp"
#include "detail/error_macros.hpp"
#include "detail/partition.hpp"
#include "shafft_internal.hpp"

#include <shafft/shafft.h>
#include <shafft/shafft_types.hpp>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mpi.h>
#include <string>
#include <vector>

#if SHAFFT_BACKEND_HIPFFT
#include <hip/hip_runtime.h>
#endif

// ---- helpers ---------------------------------------------------------------

namespace {

/// @brief Duplicate a C string onto the heap.
char* heapStrDup(const char* src, size_t len) {
  if (!src || len == 0)
    return nullptr;
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
  auto* dst = static_cast<char*>(std::malloc(len + 1));
  if (dst) {
    std::memcpy(dst, src, len);
    dst[len] = '\0';
  }
  return dst;
}

/// @brief Free a heap string and null the pointer.
void freeStr(char*& ptr, size_t& len) {
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
  std::free(ptr);
  ptr = nullptr;
  len = 0;
}

/// @brief Allocate a size_t array of n elements, zero-initialized.
size_t* allocSizeArray(int n) {
  if (n <= 0)
    return nullptr;
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
  auto* p = static_cast<size_t*>(std::calloc(static_cast<size_t>(n), sizeof(size_t)));
  return p;
}

/// @brief Allocate an int array of n elements, zero-initialized.
int* allocIntArray(int n) {
  if (n <= 0)
    return nullptr;
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
  auto* p = static_cast<int*>(std::calloc(static_cast<size_t>(n), sizeof(int)));
  return p;
}

/// @brief Free a pointer and null it.
template <typename T>
void freeArray(T*& ptr) {
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
  std::free(ptr);
  ptr = nullptr;
}

/// @brief Map C precision enum to C++ FFTType.
shafft::FFTType mapPrecision(shafft_t p) {
  return (p == SHAFFT_Z2Z) ? shafft::FFTType::Z2Z : shafft::FFTType::C2C;
}

/// @brief Map C strategy enum to C++ DecompositionStrategy.
shafft::DecompositionStrategy mapStrategy(shafft_decomposition_strategy_t s) {
  return (s == SHAFFT_MINIMIZE_NDA) ? shafft::DecompositionStrategy::MINIMIZE_NDA
                                    : shafft::DecompositionStrategy::MAXIMIZE_NDA;
}

/// @brief Compute Cartesian coordinates from a linear rank.
/// Uses C (row-major) ordering, matching MPI_Cart_create with reorder=0.
void cartCoords(int rank, int nda, const int* dims, int* coords) {
  int remaining = rank;
  for (int j = nda - 1; j >= 0; --j) {
    coords[j] = remaining % dims[j];
    remaining /= dims[j];
  }
}

/// @brief Populate an nd_layout from subsize/offset arrays.
void populateNDLayout(
    shafft_nd_layout_t& lay, int ndim, int nda, const size_t* subsize, const size_t* offset) {
  size_t elems = 1;
  for (int i = 0; i < ndim; ++i) {
    lay.subsize[i] = subsize[i];
    lay.offset[i] = offset[i];
    elems *= subsize[i];
  }
  lay.localElements = elems;
  lay.hasLocalElements = (elems > 0) ? 1 : 0;

  // Contiguous axes: indices [nda..ndim-1]
  int nca = ndim - nda;
  for (int i = 0; i < nca; ++i)
    lay.contiguousAxes[i] = nda + i;
  // Distributed axes: indices [0..nda-1]
  for (int i = 0; i < nda; ++i)
    lay.distributedAxes[i] = i;
}

} // anonymous namespace

// ---- topology --------------------------------------------------------------

namespace shafft::detail {

int queryTopology(char** hostname,
                  size_t* hostnameLen,
                  char** deviceName,
                  size_t* deviceNameLen,
                  int* nodeId,
                  int* nodeCount,
                  MPI_Comm comm) noexcept {
  try {
    if (!hostname || !hostnameLen || !deviceName || !deviceNameLen || !nodeId || !nodeCount)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);

    // -- hostname --
    char nameBuf[MPI_MAX_PROCESSOR_NAME + 1] = {};
    int nameLen = 0;
    SHAFFT_MPI_OR_FAIL(MPI_Get_processor_name(nameBuf, &nameLen));
    *hostnameLen = static_cast<size_t>(nameLen);
    *hostname = heapStrDup(nameBuf, *hostnameLen);
    if (!*hostname)
      SHAFFT_FAIL(SHAFFT_ERR_ALLOC);

    // -- node id via shared-memory communicator --
    MPI_Comm shmComm = MPI_COMM_NULL;
    SHAFFT_MPI_OR_FAIL(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmComm));

    // shm rank 0 on each node gets a unique node colour
    int shmRank = 0;
    SHAFFT_MPI_OR_FAIL(MPI_Comm_rank(shmComm, &shmRank));

    int worldRank = 0, worldSize = 0;
    SHAFFT_MPI_OR_FAIL(MPI_Comm_rank(comm, &worldRank));
    SHAFFT_MPI_OR_FAIL(MPI_Comm_size(comm, &worldSize));

    // Broadcast shm-rank-0's global rank within each node
    int nodeLeader = worldRank;
    SHAFFT_MPI_OR_FAIL(MPI_Bcast(&nodeLeader, 1, MPI_INT, 0, shmComm));

    // Gather all node leaders at root, assign contiguous IDs, scatter back
    std::vector<int> allLeaders(worldSize);
    SHAFFT_MPI_OR_FAIL(MPI_Allgather(&nodeLeader, 1, MPI_INT, allLeaders.data(), 1, MPI_INT, comm));

    // Build sorted unique leaders to assign contiguous node IDs
    std::vector<int> unique = allLeaders;
    std::sort(unique.begin(), unique.end());
    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());

    *nodeCount = static_cast<int>(unique.size());
    auto it = std::lower_bound(unique.begin(), unique.end(), nodeLeader);
    *nodeId = static_cast<int>(it - unique.begin());

    SHAFFT_MPI_OR_FAIL(MPI_Comm_free(&shmComm));

    // -- device name --
    *deviceName = nullptr;
    *deviceNameLen = 0;

#if SHAFFT_BACKEND_HIPFFT
    int deviceId = -1;
    hipError_t hrc = hipGetDevice(&deviceId);
    if (hrc == hipSuccess && deviceId >= 0) {
      hipDeviceProp_t props = {};
      hrc = hipGetDeviceProperties(&props, deviceId);
      if (hrc == hipSuccess) {
        size_t dlen = std::strlen(props.name);
        *deviceName = heapStrDup(props.name, dlen);
        *deviceNameLen = dlen;
      }
    }
#endif

    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

// ---- N-D config lifecycle --------------------------------------------------

int configNDInit(shafft_nd_config_t* cfg,
                 int ndim,
                 const size_t* globalShape,
                 shafft_t precision,
                 const int* commDims,
                 int hintNda,
                 shafft_decomposition_strategy_t strategy,
                 shafft_transform_layout_t outputPolicy,
                 size_t memLimit,
                 MPI_Comm comm) noexcept {
  try {
    if (!cfg)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);
    if (ndim < 1)
      SHAFFT_FAIL(SHAFFT_ERR_INVALID_DIM);
    if (!globalShape)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);

    // Detect already-initialized (structSize set by previous init)
    if (cfg->structSize == sizeof(shafft_nd_config_t))
      SHAFFT_FAIL(SHAFFT_ERR_INVALID_STATE);

    // Zero the struct first
    std::memset(cfg, 0, sizeof(shafft_nd_config_t));

    cfg->structSize = sizeof(shafft_nd_config_t);
    cfg->apiVersion = kConfigApiVersion;
    cfg->ndim = ndim;
    cfg->activeRank = -1;
    cfg->worldComm = MPI_COMM_NULL;
    cfg->activeComm = MPI_COMM_NULL;

    // Allocate arrays
    cfg->globalShape = allocSizeArray(ndim);
    cfg->hintCommDims = allocIntArray(ndim);
    cfg->commDims = allocIntArray(ndim);

    // Layout arrays
    cfg->initial.subsize = allocSizeArray(ndim);
    cfg->initial.offset = allocSizeArray(ndim);
    cfg->initial.contiguousAxes = allocIntArray(ndim);
    cfg->initial.distributedAxes = allocIntArray(ndim);
    cfg->output.subsize = allocSizeArray(ndim);
    cfg->output.offset = allocSizeArray(ndim);
    cfg->output.contiguousAxes = allocIntArray(ndim);
    cfg->output.distributedAxes = allocIntArray(ndim);

    // Check all allocations succeeded
    if (!cfg->globalShape || !cfg->hintCommDims || !cfg->commDims || !cfg->initial.subsize ||
        !cfg->initial.offset || !cfg->initial.contiguousAxes || !cfg->initial.distributedAxes ||
        !cfg->output.subsize || !cfg->output.offset || !cfg->output.contiguousAxes ||
        !cfg->output.distributedAxes) {
      configNDRelease(cfg);
      SHAFFT_FAIL(SHAFFT_ERR_ALLOC);
    }

    // Copy inputs
    for (int i = 0; i < ndim; ++i)
      cfg->globalShape[i] = globalShape[i];

    if (commDims) {
      for (int i = 0; i < ndim; ++i)
        cfg->hintCommDims[i] = commDims[i];
    }
    // else hintCommDims is already zero-filled (auto)

    cfg->precision = precision;
    cfg->hintNda = hintNda;
    cfg->strategy = strategy;
    cfg->outputPolicy = (outputPolicy == 0) ? SHAFFT_LAYOUT_REDISTRIBUTED : outputPolicy;
    cfg->memLimit = memLimit;

    // Store communicator (dup)
    SHAFFT_MPI_OR_FAIL(MPI_Comm_dup(comm, &cfg->worldComm));

    // Resolve decomposition using stored worldComm
    int rc = configNDResolve(cfg);
    if (rc != 0) {
      // Clean up on failure
      configNDRelease(cfg);
      return rc;
    }

    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

void configNDRelease(shafft_nd_config_t* cfg) noexcept {
  if (!cfg)
    return;
  // Guard: skip cleanup if the struct was never initialized (or already released).
  if (cfg->structSize == 0)
    return;

  freeArray(cfg->globalShape);
  freeArray(cfg->hintCommDims);
  freeArray(cfg->commDims);

  freeArray(cfg->initial.subsize);
  freeArray(cfg->initial.offset);
  freeArray(cfg->initial.contiguousAxes);
  freeArray(cfg->initial.distributedAxes);
  freeArray(cfg->output.subsize);
  freeArray(cfg->output.offset);
  freeArray(cfg->output.contiguousAxes);
  freeArray(cfg->output.distributedAxes);

  freeStr(cfg->hostname, cfg->hostnameLen);
  freeStr(cfg->deviceName, cfg->deviceNameLen);

  // Free owned communicators
  if (cfg->activeComm != MPI_COMM_NULL)
    MPI_Comm_free(&cfg->activeComm);
  if (cfg->worldComm != MPI_COMM_NULL)
    MPI_Comm_free(&cfg->worldComm);

  std::memset(cfg, 0, sizeof(shafft_nd_config_t));
}

int configNDResolve(shafft_nd_config_t* cfg) noexcept {
  try {
    if (!cfg)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);
    if (cfg->structSize != sizeof(shafft_nd_config_t))
      SHAFFT_FAIL(SHAFFT_ERR_INVALID_STATE); // Not initialized
    if (cfg->ndim < 1)
      SHAFFT_FAIL(SHAFFT_ERR_INVALID_DIM);
    if (cfg->worldComm == MPI_COMM_NULL)
      SHAFFT_FAIL(SHAFFT_ERR_INVALID_COMM);

    MPI_Comm comm = cfg->worldComm;

    const int ndim = cfg->ndim;

    // Validate globalShape
    for (int i = 0; i < ndim; ++i) {
      if (cfg->globalShape[i] == 0)
        SHAFFT_FAIL(SHAFFT_ERR_INVALID_DIM);
      if (cfg->globalShape[i] > static_cast<size_t>(std::numeric_limits<int>::max()))
        SHAFFT_FAIL(SHAFFT_ERR_SIZE_OVERFLOW);
    }

    // Validate precision
    if (cfg->precision != SHAFFT_C2C && cfg->precision != SHAFFT_Z2Z)
      SHAFFT_FAIL(SHAFFT_ERR_INVALID_FFTTYPE);

    // Clear resolved flag
    cfg->flags &= ~SHAFFT_CONFIG_RESOLVED;

    // Convert globalShape to int for internal API
    std::vector<int> intShape(ndim);
    for (int i = 0; i < ndim; ++i)
      intShape[i] = static_cast<int>(cfg->globalShape[i]);

    // Copy hints into working buffers (configurationND modifies commDims/nda in-place)
    std::vector<int> workCommDims(ndim);
    for (int i = 0; i < ndim; ++i)
      workCommDims[i] = cfg->hintCommDims[i];
    int workNda = cfg->hintNda;

    // Store original hints for change detection
    std::vector<int> origCommDims = workCommDims;
    int origNda = workNda;

    // Resolve decomposition
    std::vector<int> intSubsize(ndim), intOffset(ndim);
    int commSize = 0;
    int rc = configurationND(ndim,
                             intShape.data(),
                             mapPrecision(cfg->precision),
                             workCommDims.data(),
                             &workNda,
                             intSubsize.data(),
                             intOffset.data(),
                             &commSize,
                             mapStrategy(cfg->strategy),
                             cfg->memLimit,
                             comm);
    if (rc != 0) {
      cfg->status = rc;
      return rc;
    }

    // Store resolved decomposition
    cfg->nda = workNda;
    cfg->commSize = commSize;
    for (int i = 0; i < ndim; ++i)
      cfg->commDims[i] = workCommDims[i];

    // Need worldRank/worldSize for output layout and activity metadata.
    int worldRank = 0, worldSize = 0;
    SHAFFT_MPI_OR_FAIL(MPI_Comm_rank(comm, &worldRank));
    SHAFFT_MPI_OR_FAIL(MPI_Comm_size(comm, &worldSize));

    // Populate initial layout
    std::vector<size_t> sizeSubsize(ndim), sizeOffset(ndim);
    for (int i = 0; i < ndim; ++i) {
      sizeSubsize[i] = static_cast<size_t>(intSubsize[i]);
      sizeOffset[i] = static_cast<size_t>(intOffset[i]);
    }
    populateNDLayout(cfg->initial, ndim, workNda, sizeSubsize.data(), sizeOffset.data());

    // Populate output layout.
    // Rule: the slab redistribution algorithm performs nda axis exchanges that
    // shift distributed axis j to position (j + nca).  After all exchanges the
    // final distributed axes are [nca, nca+1, ..., ndim-1], each partitioned by
    // the same communicator dimension j that initially partitioned axis j.
    // Axes [0..nca-1] become contiguous (full, undistributed).
    //
    // For INITIAL output policy (or nda==0), output == initial.
    if (workNda == 0 || cfg->outputPolicy == SHAFFT_LAYOUT_INITIAL) {
      populateNDLayout(cfg->output, ndim, workNda, sizeSubsize.data(), sizeOffset.data());
    } else {
      int nca = ndim - workNda;

      // Compute Cartesian coordinates (same rank mapping as MPI_Cart_create
      // with reorder=0 on the first commSize ranks).
      std::vector<int> coords(workNda);
      if (worldRank < commSize)
        cartCoords(worldRank, workNda, workCommDims.data(), coords.data());

      size_t elems = 1;
      for (int k = 0; k < ndim; ++k) {
        if (k < nca) {
          // Originally distributed, now contiguous (full extent).
          cfg->output.subsize[k] = cfg->globalShape[k];
          cfg->output.offset[k] = 0;
        } else {
          // Originally contiguous, now distributed by commDims[k - nca].
          int j = k - nca;
          auto bp =
              block_partition(static_cast<int>(cfg->globalShape[k]), workCommDims[j], coords[j]);
          cfg->output.subsize[k] = static_cast<size_t>(bp.localN);
          cfg->output.offset[k] = static_cast<size_t>(bp.localStart);
        }
        elems *= cfg->output.subsize[k];
      }
      cfg->output.localElements = elems;
      cfg->output.hasLocalElements = (elems > 0) ? 1 : 0;

      // Output contiguous axes: [0..nca-1]
      for (int i = 0; i < nca; ++i)
        cfg->output.contiguousAxes[i] = i;
      // Output distributed axes: [nca..ndim-1]
      for (int i = 0; i < workNda; ++i)
        cfg->output.distributedAxes[i] = nca + i;
    }

    // Allocation elements: max of initial and output.
    // NOTE: intermediate exchange stages may be larger; plan.allocSize() is
    // authoritative once the plan is created.
    cfg->allocElements = std::max(cfg->initial.localElements, cfg->output.localElements);

    // Activity metadata
    cfg->isActive = (cfg->initial.localElements > 0) ? 1 : 0;

    // Split into active / inactive subsets
    // Free old activeComm if re-resolving
    if (cfg->activeComm != MPI_COMM_NULL) {
      SHAFFT_MPI_OR_FAIL(MPI_Comm_free(&cfg->activeComm));
      cfg->activeComm = MPI_COMM_NULL;
    }

    SHAFFT_MPI_OR_FAIL(
        MPI_Comm_split(comm, cfg->isActive ? 0 : MPI_UNDEFINED, worldRank, &cfg->activeComm));

    if (cfg->activeComm != MPI_COMM_NULL) {
      SHAFFT_MPI_OR_FAIL(MPI_Comm_rank(cfg->activeComm, &cfg->activeRank));
      SHAFFT_MPI_OR_FAIL(MPI_Comm_size(cfg->activeComm, &cfg->activeSize));
    } else {
      cfg->activeRank = -1;
      cfg->activeSize = 0;
    }
    // Broadcast activeSize to inactive ranks
    SHAFFT_MPI_OR_FAIL(MPI_Bcast(&cfg->activeSize, 1, MPI_INT, 0, comm));

    // Topology metadata
    // Free any previous strings
    freeStr(cfg->hostname, cfg->hostnameLen);
    freeStr(cfg->deviceName, cfg->deviceNameLen);

    rc = queryTopology(&cfg->hostname,
                       &cfg->hostnameLen,
                       &cfg->deviceName,
                       &cfg->deviceNameLen,
                       &cfg->nodeId,
                       &cfg->nodeCount,
                       comm);
    if (rc != 0) {
      cfg->status = rc;
      return rc;
    }

    // Set flags
    cfg->flags = SHAFFT_CONFIG_RESOLVED;
    if (origCommDims != workCommDims)
      cfg->flags |= SHAFFT_CONFIG_CHANGED_COMM_DIMS;
    if (origNda != workNda)
      cfg->flags |= SHAFFT_CONFIG_CHANGED_NDA;
    if (cfg->activeSize < worldSize)
      cfg->flags |= SHAFFT_CONFIG_INACTIVE_RANKS;

    cfg->status = SHAFFT_STATUS(SHAFFT_SUCCESS);
    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

// ---- 1-D config lifecycle --------------------------------------------------

int config1DInit(shafft_1d_config_t* cfg,
                 size_t globalSize,
                 shafft_t precision,
                 MPI_Comm comm) noexcept {
  try {
    if (!cfg)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);
    if (globalSize == 0)
      SHAFFT_FAIL(SHAFFT_ERR_INVALID_DIM);
    if (globalSize > static_cast<size_t>(std::numeric_limits<int>::max()))
      SHAFFT_FAIL(SHAFFT_ERR_SIZE_OVERFLOW);
    if (precision != SHAFFT_C2C && precision != SHAFFT_Z2Z)
      SHAFFT_FAIL(SHAFFT_ERR_INVALID_FFTTYPE);

    // Detect already-initialized
    if (cfg->structSize == sizeof(shafft_1d_config_t))
      SHAFFT_FAIL(SHAFFT_ERR_INVALID_STATE);

    std::memset(cfg, 0, sizeof(shafft_1d_config_t));

    cfg->structSize = sizeof(shafft_1d_config_t);
    cfg->apiVersion = kConfigApiVersion;
    cfg->globalSize = globalSize;
    cfg->precision = precision;
    cfg->activeRank = -1;
    cfg->worldComm = MPI_COMM_NULL;
    cfg->activeComm = MPI_COMM_NULL;

    // Store communicator (dup)
    SHAFFT_MPI_OR_FAIL(MPI_Comm_dup(comm, &cfg->worldComm));

    // Resolve using stored worldComm
    int rc = config1DResolve(cfg);
    if (rc != 0) {
      config1DRelease(cfg);
      return rc;
    }

    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

void config1DRelease(shafft_1d_config_t* cfg) noexcept {
  if (!cfg)
    return;
  // Guard: skip cleanup if the struct was never initialized (or already released).
  if (cfg->structSize == 0)
    return;

  freeStr(cfg->hostname, cfg->hostnameLen);
  freeStr(cfg->deviceName, cfg->deviceNameLen);

  // Free owned communicators
  if (cfg->activeComm != MPI_COMM_NULL)
    MPI_Comm_free(&cfg->activeComm);
  if (cfg->worldComm != MPI_COMM_NULL)
    MPI_Comm_free(&cfg->worldComm);

  std::memset(cfg, 0, sizeof(shafft_1d_config_t));
}

int config1DResolve(shafft_1d_config_t* cfg) noexcept {
  try {
    if (!cfg)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);
    if (cfg->structSize != sizeof(shafft_1d_config_t))
      SHAFFT_FAIL(SHAFFT_ERR_INVALID_STATE);
    if (cfg->globalSize == 0)
      SHAFFT_FAIL(SHAFFT_ERR_INVALID_DIM);
    if (cfg->globalSize > static_cast<size_t>(std::numeric_limits<int>::max()))
      SHAFFT_FAIL(SHAFFT_ERR_SIZE_OVERFLOW);
    if (cfg->worldComm == MPI_COMM_NULL)
      SHAFFT_FAIL(SHAFFT_ERR_INVALID_COMM);

    MPI_Comm comm = cfg->worldComm;

    cfg->flags &= ~SHAFFT_CONFIG_RESOLVED;

    // Query layout from backend
    size_t localN = 0, localStart = 0, localAllocSize = 0;
    int rc = configuration1D(
        cfg->globalSize, &localN, &localStart, &localAllocSize, mapPrecision(cfg->precision), comm);
    if (rc != 0) {
      cfg->status = rc;
      return rc;
    }

    // Initial layout
    cfg->initial.localSize = localN;
    cfg->initial.localStart = localStart;
    cfg->initial.hasLocalElements = (localN > 0) ? 1 : 0;

    // Output layout (same as initial for 1-D slab)
    cfg->output.localSize = localN;
    cfg->output.localStart = localStart;
    cfg->output.hasLocalElements = (localN > 0) ? 1 : 0;

    cfg->allocElements = localAllocSize;

    // Activity
    cfg->isActive = (localN > 0) ? 1 : 0;

    // Active rank/size
    int worldRank = 0, worldSize = 0;
    SHAFFT_MPI_OR_FAIL(MPI_Comm_rank(comm, &worldRank));
    SHAFFT_MPI_OR_FAIL(MPI_Comm_size(comm, &worldSize));

    // Free old activeComm if re-resolving
    if (cfg->activeComm != MPI_COMM_NULL) {
      SHAFFT_MPI_OR_FAIL(MPI_Comm_free(&cfg->activeComm));
      cfg->activeComm = MPI_COMM_NULL;
    }

    SHAFFT_MPI_OR_FAIL(
        MPI_Comm_split(comm, cfg->isActive ? 0 : MPI_UNDEFINED, worldRank, &cfg->activeComm));

    if (cfg->activeComm != MPI_COMM_NULL) {
      SHAFFT_MPI_OR_FAIL(MPI_Comm_rank(cfg->activeComm, &cfg->activeRank));
      SHAFFT_MPI_OR_FAIL(MPI_Comm_size(cfg->activeComm, &cfg->activeSize));
    } else {
      cfg->activeRank = -1;
      cfg->activeSize = 0;
    }
    SHAFFT_MPI_OR_FAIL(MPI_Bcast(&cfg->activeSize, 1, MPI_INT, 0, comm));

    // Topology
    freeStr(cfg->hostname, cfg->hostnameLen);
    freeStr(cfg->deviceName, cfg->deviceNameLen);

    rc = queryTopology(&cfg->hostname,
                       &cfg->hostnameLen,
                       &cfg->deviceName,
                       &cfg->deviceNameLen,
                       &cfg->nodeId,
                       &cfg->nodeCount,
                       comm);
    if (rc != 0) {
      cfg->status = rc;
      return rc;
    }

    // Flags
    cfg->flags = SHAFFT_CONFIG_RESOLVED;
    if (cfg->activeSize < worldSize)
      cfg->flags |= SHAFFT_CONFIG_INACTIVE_RANKS;

    cfg->status = SHAFFT_STATUS(SHAFFT_SUCCESS);
    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

} // namespace shafft::detail

// ---- C API wrappers (extern "C") -------------------------------------------
// These delegate to the detail:: functions and are linked into both shafftc++
// and the C shared library.

extern "C" {

int shafftConfigNDInit(shafft_nd_config_t* cfg,
                       int ndim,
                       const size_t* globalShape,
                       shafft_t precision,
                       const int* commDims,
                       int hintNda,
                       shafft_decomposition_strategy_t strategy,
                       shafft_transform_layout_t outputPolicy,
                       size_t memLimit,
                       MPI_Comm comm) {
  return shafft::detail::configNDInit(
      cfg, ndim, globalShape, precision, commDims, hintNda, strategy, outputPolicy, memLimit, comm);
}

void shafftConfigNDRelease(shafft_nd_config_t* cfg) {
  shafft::detail::configNDRelease(cfg);
}

int shafftConfigNDResolve(shafft_nd_config_t* cfg) {
  return shafft::detail::configNDResolve(cfg);
}

int shafftConfig1DInit(shafft_1d_config_t* cfg,
                       size_t globalSize,
                       shafft_t precision,
                       MPI_Comm comm) {
  return shafft::detail::config1DInit(cfg, globalSize, precision, comm);
}

void shafftConfig1DRelease(shafft_1d_config_t* cfg) {
  shafft::detail::config1DRelease(cfg);
}

int shafftConfig1DResolve(shafft_1d_config_t* cfg) {
  return shafft::detail::config1DResolve(cfg);
}

} // extern "C"
