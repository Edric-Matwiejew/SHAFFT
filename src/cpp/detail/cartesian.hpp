/**
 * @file cartesian.hpp
 * @brief MPI Cartesian grid and subcommunicator utilities.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef SHAFFT_DETAIL_CARTESIAN_HPP
#define SHAFFT_DETAIL_CARTESIAN_HPP

#include <mpi.h>
#include <vector>

namespace shafft::detail {

/**
 * @brief Create MPI subcommunicators from a Cartesian communicator.
 *
 * For each dimension i in [0, ndims), creates a subcommunicator containing
 * all ranks along that dimension (i.e., with remain_dims[i] = 1).
 *
 * @param cartComm   Input Cartesian communicator (must be valid, not MPI_COMM_NULL)
 * @param ndims      Number of dimensions in the Cartesian grid
 * @param subcomms   [out] Array of size ndims to receive subcommunicators
 *
 * @note Caller is responsible for freeing the subcommunicators with MPI_Comm_free.
 */
inline void make_subcomms(MPI_Comm cartComm, int ndims, MPI_Comm* subcomms) {
  std::vector<int> remdims(ndims, 0);
  for (int i = 0; i < ndims; ++i) {
    remdims[i] = 1;
    MPI_Cart_sub(cartComm, remdims.data(), &subcomms[i]);
    remdims[i] = 0;
  }
}

/**
 * @brief Create a Cartesian communicator with auto-computed dimensions.
 *
 * Uses MPI_Dims_create to compute balanced dimensions, then creates
 * a Cartesian communicator with non-periodic boundaries.
 *
 * @param comm       Input communicator
 * @param ndims      Number of dimensions for the Cartesian grid
 * @param reorder    Whether to allow rank reordering (0 or 1)
 * @param cartDims   [out] Array of size ndims to receive computed dimensions
 * @param cartComm   [out] The created Cartesian communicator
 *
 * @note cartDims array is zeroed before calling MPI_Dims_create.
 * @note Caller is responsible for freeing cartComm with MPI_Comm_free.
 */
inline void make_cart(MPI_Comm comm, int ndims, int reorder, int* cartDims, MPI_Comm* cartComm) {
  int nprocs = 0;
  MPI_Comm_size(comm, &nprocs);

  std::vector<int> periods(ndims, 0);
  for (int i = 0; i < ndims; ++i) {
    cartDims[i] = 0;
  }

  MPI_Dims_create(nprocs, ndims, cartDims);
  MPI_Cart_create(comm, ndims, cartDims, periods.data(), reorder, cartComm);
}

/**
 * @brief Create a Cartesian communicator and subcommunicators in one call.
 *
 * Combines make_cart and make_subcomms. The intermediate Cartesian communicator
 * is freed after creating the subcommunicators.
 *
 * @param comm       Input communicator
 * @param ndims      Number of dimensions for the Cartesian grid
 * @param subcomms   [out] Array of size ndims to receive subcommunicators
 *
 * @note Caller is responsible for freeing subcommunicators with MPI_Comm_free.
 */
inline void make_cart_subcomms(MPI_Comm comm, int ndims, MPI_Comm* subcomms) {
  std::vector<int> cartDims(ndims, 0);
  MPI_Comm cartComm = MPI_COMM_NULL;

  make_cart(comm, ndims, /*reorder=*/1, cartDims.data(), &cartComm);
  make_subcomms(cartComm, ndims, subcomms);
  MPI_Comm_free(&cartComm);
}

/**
 * @brief Create a Cartesian communicator and subcommunicators, returning dimensions.
 *
 * Same as make_cart_subcomms but also returns the computed Cartesian dimensions.
 *
 * @param comm       Input communicator
 * @param ndims      Number of dimensions for the Cartesian grid
 * @param cartDims   [out] Array of size ndims to receive computed dimensions
 * @param subcomms   [out] Array of size ndims to receive subcommunicators
 *
 * @note Caller is responsible for freeing subcommunicators with MPI_Comm_free.
 */
inline void make_cart_subcomms(MPI_Comm comm, int ndims, int* cartDims, MPI_Comm* subcomms) {
  MPI_Comm cartComm = MPI_COMM_NULL;

  make_cart(comm, ndims, /*reorder=*/1, cartDims, &cartComm);
  make_subcomms(cartComm, ndims, subcomms);
  MPI_Comm_free(&cartComm);
}

} // namespace shafft::detail

#endif // SHAFFT_DETAIL_CARTESIAN_HPP
