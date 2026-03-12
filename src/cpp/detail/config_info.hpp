/// @file config_info.hpp
/// @brief Internal helpers for configuration object lifecycle and resolution.
/// @details Shared by C and C++ front-ends. Not for library consumers.

#ifndef SHAFFT_SRC_CONFIG_INFO_HPP
#define SHAFFT_SRC_CONFIG_INFO_HPP

#include <shafft/shafft.h>

#include <cstddef>
#include <mpi.h>

namespace shafft::detail {

// ---- API version -----------------------------------------------------------

/// @brief Current config API version. Incremented when struct layout changes.
constexpr int kConfigApiVersion = 1;

// ---- N-D config lifecycle --------------------------------------------------

/// @brief Initialize and resolve an N-D config struct (collective).
/// @pre cfg is zero-initialized. ndim >= 1.
/// @post All pointer fields allocated; decomposition resolved; comms owned.
int configNDInit(shafft_nd_config_t* cfg,
                 int ndim,
                 const size_t* globalShape,
                 shafft_t precision,
                 const int* commDims,
                 int hintNda,
                 shafft_decomposition_strategy_t strategy,
                 shafft_transform_layout_t outputPolicy,
                 size_t memLimit,
                 MPI_Comm comm) noexcept;

/// @brief Free internal arrays/strings in an N-D config struct and zero-fill.
void configNDRelease(shafft_nd_config_t* cfg) noexcept;

/// @brief Re-resolve an N-D config struct (collective on stored worldComm).
/// @details Frees old activeComm, re-computes decomposition, creates new activeComm.
int configNDResolve(shafft_nd_config_t* cfg) noexcept;

// ---- 1-D config lifecycle --------------------------------------------------

/// @brief Initialize and resolve a 1-D config struct (collective).
int config1DInit(shafft_1d_config_t* cfg,
                 size_t globalSize,
                 shafft_t precision,
                 MPI_Comm comm) noexcept;

/// @brief Free internal strings in a 1-D config struct and zero-fill.
void config1DRelease(shafft_1d_config_t* cfg) noexcept;

/// @brief Re-resolve a 1-D config struct (collective on stored worldComm).
int config1DResolve(shafft_1d_config_t* cfg) noexcept;

// ---- Topology helpers ------------------------------------------------------

/// @brief Populate hostname, deviceName, nodeId, nodeCount for a config.
/// @details Calls MPI_Get_processor_name and computes shared-memory-based
///          node IDs. Device name is queried from HIP runtime if available.
/// @param[out] hostname      Heap-allocated hostname string.
/// @param[out] hostnameLen   Length of hostname.
/// @param[out] deviceName    Heap-allocated device name (NULL if no device).
/// @param[out] deviceNameLen Length of device name.
/// @param[out] nodeId        Node index (0-based).
/// @param[out] nodeCount     Total distinct nodes.
/// @param      comm          MPI communicator.
int queryTopology(char** hostname,
                  size_t* hostnameLen,
                  char** deviceName,
                  size_t* deviceNameLen,
                  int* nodeId,
                  int* nodeCount,
                  MPI_Comm comm) noexcept;

} // namespace shafft::detail

#endif // SHAFFT_SRC_CONFIG_INFO_HPP
