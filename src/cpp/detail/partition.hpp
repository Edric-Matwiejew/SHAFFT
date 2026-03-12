/**
 * @file partition.hpp
 * @brief Block partitioning utilities for distributed array decomposition.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef SHAFFT_DETAIL_PARTITION_HPP
#define SHAFFT_DETAIL_PARTITION_HPP

#include <algorithm>

namespace shafft::detail {

/**
 * @brief Result of a 1D block partition computation.
 */
struct BlockPartition {
  int localN;     ///< Number of elements assigned to this rank
  int localStart; ///< Starting index for this rank's block
};

/**
 * @brief Compute a 1D block partition using quotient/remainder distribution.
 *
 * Distributes `total` elements across `nparts` partitions. Lower-ranked
 * partitions receive one extra element when there is a remainder.
 *
 * @param total   Total number of elements to distribute
 * @param nparts  Number of partitions (e.g., MPI ranks)
 * @param rank    Rank of the partition to compute (0-indexed)
 * @return BlockPartition containing localN and localStart
 *
 * @note For total=10, nparts=4:
 *   - rank 0: localN=3, localStart=0  (elements 0-2)
 *   - rank 1: localN=3, localStart=3  (elements 3-5)
 *   - rank 2: localN=2, localStart=6  (elements 6-7)
 *   - rank 3: localN=2, localStart=8  (elements 8-9)
 */
inline BlockPartition block_partition(int total, int nparts, int rank) noexcept {
  const int q = total / nparts;
  const int r = total % nparts;
  return {q + (rank < r ? 1 : 0), (rank < r) ? rank * (q + 1) : r * (q + 1) + (rank - r) * q};
}

/**
 * @brief Compute block partition and write to output pointers.
 *
 * Convenience overload for code that uses output parameters.
 *
 * @param total      Total number of elements to distribute
 * @param nparts     Number of partitions
 * @param rank       Rank of the partition to compute
 * @param localN     [out] Number of elements for this rank
 * @param localStart [out] Starting index for this rank
 */
inline void
block_partition(int total, int nparts, int rank, int* localN, int* localStart) noexcept {
  auto bp = block_partition(total, nparts, rank);
  *localN = bp.localN;
  *localStart = bp.localStart;
}

} // namespace shafft::detail

#endif // SHAFFT_DETAIL_PARTITION_HPP
