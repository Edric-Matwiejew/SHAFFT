#include "detail/array_utils.hpp"
#include "detail/error_macros.hpp"
#include "detail/partition.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mpi.h>
#include <vector>

// ---- helpers ---------------------------------------------------------------

static bool withinIntLimitExact(int ndim, const int dims[]) noexcept {
  long double prod = 1.0L;
  for (int i = 0; i < ndim; ++i) {
    if (dims[i] <= 0)
      return false;
    prod *= static_cast<long double>(dims[i]);
    if (prod > static_cast<long double>(std::numeric_limits<int32_t>::max()))
      return false;
  }
  return true;
}

// Check if a local block size fits within the memory limit.
// memLimit == 0 means unlimited (always returns true).
static bool withinMemLimitGuarded(int ndim,
                                  const int sizePerRank[],
                                  shafft::FFTType precision,
                                  size_t memLimit) noexcept {
  if (memLimit == 0)
    return true;

  size_t typeSize = 0;
  switch (precision) {
  case shafft::FFTType::C2C:
    typeSize = 2 * sizeof(float);
    break;
  case shafft::FFTType::Z2Z:
    typeSize = 2 * sizeof(double);
    break;
  default:
    return false;
  }

  // Each rank holds both data and work buffers concurrently.
  const size_t bufferCount = 2;

  __uint128_t elems = 1;
  for (int i = 0; i < ndim; ++i) {
    if (sizePerRank[i] <= 0)
      return false;
    elems *= static_cast<uint64_t>(sizePerRank[i]);
    if (elems > ((__uint128_t)memLimit) / (bufferCount * typeSize))
      return false;
  }
  return (elems * bufferCount * typeSize) <= memLimit;
}

static int allreduceBoolAnd(bool localOk, bool* globalOk, MPI_Comm comm) noexcept {
  if (!globalOk)
    return SHAFFT_STATUS(SHAFFT_ERR_NULLPTR);

  *globalOk = false;

  bool reduced = false;

  SHAFFT_MPI_OR_FAIL(MPI_Allreduce(&localOk, &reduced, 1, MPI_C_BOOL, MPI_LAND, comm));

  *globalOk = reduced;
  return SHAFFT_STATUS(SHAFFT_SUCCESS);
}

static bool allZero(const int* a, int n) {
  for (int i = 0; i < n; ++i)
    if (a[i] != 0)
      return false;
  return true;
}

// Cross-axis constraint: commDims[i] <= min(sizes[i], sizes[i + nca]) for i < d
// where nca = ndim - d. During exchanges, distributed axis i will partition
// contiguous axis (i + nca), so the communicator size must not exceed that axis's size.
static void calcMinCaps(int ndim, int d, const int* sizes, std::vector<int>& caps) {
  caps.assign(d, 1);
  int nca = ndim - d;
  for (int i = 0; i < d; ++i)
    caps[i] = std::min(sizes[i], sizes[i + nca]);
}

// Create a grid size for a Cartesian communicator for a tensor with nda distributed axes.
// Uses MPI_Dims_create for balanced factor distribution, then caps by cross-axis constraints.
// Returns 0 on success, non-zero on failure.
static int
createCommDims(int ndim, int commSize, int nda, const int* sizes, int* commDims) noexcept {
  // Initialize: all dimensions set to 1
  std::fill(commDims, commDims + ndim, 1);

  if (nda <= 0 || nda > ndim)
    return 1;

  if (nda == 1) {
    // Single distributed axis: cap by axis size
    commDims[0] = std::min(commSize, sizes[0]);
    return 0;
  }

  // Compute per-axis caps implied by redistribution
  std::vector<int> caps;
  calcMinCaps(ndim, nda, sizes, caps);

  // Use MPI_Dims_create for balanced factor distribution
  std::vector<int> mpiDims(nda, 0);
  MPI_Dims_create(commSize, nda, mpiDims.data());

  // Apply caps: each commDims[i] = min(mpiDims[i], caps[i])
  for (int i = 0; i < nda; ++i) {
    commDims[i] = std::min(mpiDims[i], caps[i]);
    if (commDims[i] < 1)
      commDims[i] = 1;
  }

  // Success criteria: non-zero product
  int prod = shafft::detail::prodN(commDims, nda);
  if (prod <= 0)
    return 2;

  return 0;
}

// Largest divisor of n that is <= limit (>=2 if possible, otherwise 1)
static int largestDivisorLeq(int n, int limit) {
  if (n <= 1 || limit < 2)
    return 1;
  int best = 1;
  int up = std::min(n, limit);
  for (int d = 2; d * d <= n; ++d) {
    if (n % d == 0) {
      int d1 = d;
      int d2 = n / d;
      if (d1 <= up)
        best = std::max(best, d1);
      if (d2 <= up)
        best = std::max(best, d2);
      if (best == up)
        return best; // can't do better than limit
    }
  }
  // If none of the proper factors fit, check n itself
  if (n <= up)
    best = std::max(best, n);
  return best;
}

// Pack candidate commDims to a left prefix with "spill", respecting caps/sizes.
// We treat each >1 entry as a multiplicative factor that can be split across
// axes using integer divisors. We keep the last tensor axis contiguous.
// Returns d' = number of leading >1 entries after packing (0..ndim-1).
static int packPrefixWithSpill(int ndim, int* grid, const int* size, const int* caps /*nullable*/) {
  // Collect multiplicative factors in order
  std::vector<int> factors;
  factors.reserve(ndim);
  for (int i = 0; i < ndim; ++i)
    if (grid[i] > 1)
      factors.push_back(grid[i]);

  // Reset grid to 1s; we'll rebuild the prefix
  for (int i = 0; i < ndim; ++i)
    grid[i] = 1;

  const int maxAxes = std::max(0, ndim - 1); // keep last axis contiguous
  int axis = 0;

  // Remaining capacity per axis (start at <= cap/size)
  auto axisCap = [&](int i) -> int {
    int c = size[i];
    if (caps)
      c = std::min(c, caps[i]);
    return std::max(1, c);
  };

  std::vector<int> capLeft(maxAxes, 1);
  for (int i = 0; i < maxAxes; ++i)
    capLeft[i] = axisCap(i);

  for (int f : factors) {
    int rem = f;
    while (rem > 1 && axis < maxAxes) {
      // how much more we can multiply on this axis
      int room = capLeft[axis] / grid[axis];
      if (room < 2) {
        ++axis;
        continue;
      }
      int used = largestDivisorLeq(rem, room);
      if (used <= 1) {
        ++axis;
        continue;
      }
      grid[axis] *= used;
      rem /= used;
      // If we filled this axis up, move on; else we may continue placing here
      if (grid[axis] == capLeft[axis])
        ++axis;
    }
    // If rem > 1 here, we couldn't place all ranks; they'll become inactive.
  }

  // Count leading >1 (d')
  int dPrime = 0;
  while (dPrime < maxAxes && grid[dPrime] > 1)
    ++dPrime;

  // Ensure remaining entries are 1 (including last axis)
  for (int i = dPrime; i < ndim; ++i)
    grid[i] = 1;
  return dPrime;
}

// Check if the local block passes int-overflow and memory limits.
// memLimit == 0 means unlimited.
static bool passesLimits(int ndim,
                         const std::vector<int>& subsize,
                         shafft::FFTType precision,
                         size_t memLimit) {
  if (!withinIntLimitExact(ndim, subsize.data()))
    return false;
  if (!withinMemLimitGuarded(ndim, subsize.data(), precision, memLimit))
    return false;
  return true;
}

static int computeLocalBlockCart(int ndim,
                                 const int* globalSize,
                                 int d,
                                 const int* grid,
                                 MPI_Comm comm,
                                 std::vector<int>& subsize,
                                 std::vector<int>& offset,
                                 MPI_Comm* cartOut) noexcept {
  subsize.assign(ndim, 0);
  offset.assign(ndim, 0);
  if (!cartOut)
    return SHAFFT_STATUS(SHAFFT_ERR_NULLPTR);
  *cartOut = MPI_COMM_NULL;

  if (d <= 0) {
    for (int i = 0; i < ndim; ++i) {
      subsize[i] = globalSize[i];
      offset[i] = 0;
    }
    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }

  int cartSize = shafft::detail::prodN(grid, d);

  int worldRank = 0;
  SHAFFT_MPI_OR_FAIL(MPI_Comm_rank(comm, &worldRank));

  MPI_Comm sub = MPI_COMM_NULL;
  const int color = (worldRank < cartSize) ? 1 : MPI_UNDEFINED;
  SHAFFT_MPI_OR_FAIL(MPI_Comm_split(comm, color, worldRank, &sub));

  // Ranks not in subcomm: no work; success with cartOut = MPI_COMM_NULL
  if (sub == MPI_COMM_NULL)
    return SHAFFT_STATUS(SHAFFT_SUCCESS);

  std::vector<int> periods(d, 0);
  SHAFFT_MPI_OR_FAIL(
      MPI_Cart_create(sub, d, const_cast<int*>(grid), periods.data(), /*reorder=*/0, cartOut));
  // Free the intermediate subcomm
  SHAFFT_MPI_OR_FAIL(MPI_Comm_free(&sub));

  int cartRank = -1;
  SHAFFT_MPI_OR_FAIL(MPI_Comm_rank(*cartOut, &cartRank));

  std::vector<int> coords(d, 0);
  SHAFFT_MPI_OR_FAIL(MPI_Cart_coords(*cartOut, cartRank, d, coords.data()));

  for (int i = 0; i < d; ++i) {
    auto bp = shafft::detail::block_partition(globalSize[i], grid[i], coords[i]);
    subsize[i] = bp.localN;
    offset[i] = bp.localStart;
  }
  for (int i = d; i < ndim; ++i) {
    subsize[i] = globalSize[i];
    offset[i] = 0;
  }

  return SHAFFT_STATUS(SHAFFT_SUCCESS);
}

// ---- internal configuration functions --------------------------------------

// Try to achieve exactly `*nda` distributed axes (strict mode) or auto-select
// the best nda based on preferMinNd (auto mode when *nda == 0).
// memLimit == 0 means unlimited.
static int configurationNDA(int ndim,
                            int size[],
                            int* nda,
                            int* subsizeOut,
                            int* offsetOut,
                            int* commDims,
                            shafft::FFTType precision,
                            size_t memLimit,
                            MPI_Comm comm,
                            bool preferMinNd) noexcept {
  try {
    int worldSize = 0;
    SHAFFT_MPI_OR_FAIL(MPI_Comm_size(comm, &worldSize));

    const bool autoMode = (*nda == 0);
    const bool wantMax = !preferMinNd; // false => search smallest d first

    int bestD = -1;
    std::vector<int> bestSubsize(ndim), bestOffset(ndim);

    auto tryD = [&](int d, bool strict) -> int {
      if (d < 1 || d > ndim - 1)
        return SHAFFT_STATUS(SHAFFT_ERR_INVALID_DECOMP);
      int rc = createCommDims(ndim, worldSize, d, size, commDims);
      if (rc != SHAFFT_STATUS(SHAFFT_SUCCESS))
        return rc;

      // In explicit/strict mode, all requested distributed axes must be >1.
      // If MPI_Dims_create (via createCommDims) could not populate every axis,
      // reject the decomposition instead of silently reducing nda.
      if (strict) {
        int gt1 = 0;
        for (int i = 0; i < d; ++i)
          if (commDims[i] > 1)
            ++gt1;
        if (gt1 != d)
          return SHAFFT_STATUS(SHAFFT_ERR_INVALID_DECOMP);
      }

      std::vector<int> caps;
      calcMinCaps(ndim, d, size, caps);

      int dEffective = d;
      if (!strict) {
        // In auto mode, allow packing to consolidate the grid for efficiency
        dEffective = packPrefixWithSpill(ndim, commDims, size, caps.data());
        if (dEffective == 0 && worldSize > 1)
          return SHAFFT_STATUS(SHAFFT_ERR_INVALID_DECOMP);

        // Recompute caps for the effective d after packing
        if (dEffective != d) {
          calcMinCaps(ndim, dEffective, size, caps);
          // Verify the packed grid satisfies the cross-axis constraint
          for (int i = 0; i < dEffective; ++i) {
            if (commDims[i] > caps[i])
              return SHAFFT_STATUS(SHAFFT_ERR_INVALID_DECOMP);
          }
        }
      }
      // In strict mode, use the grid from createCommDims directly (no packing)

      std::vector<int> subsize, offset;
      MPI_Comm cart = MPI_COMM_NULL;
      rc = computeLocalBlockCart(ndim, size, dEffective, commDims, comm, subsize, offset, &cart);
      if (rc != SHAFFT_STATUS(SHAFFT_SUCCESS))
        return rc;

      bool localOk = (cart == MPI_COMM_NULL) || passesLimits(ndim, subsize, precision, memLimit);
      bool globalOk = false;
      rc = allreduceBoolAnd(localOk, &globalOk, comm);
      if (cart != MPI_COMM_NULL) {
        int freeRc = MPI_Comm_free(&cart);
        if (freeRc != MPI_SUCCESS) {
          shafft::detail::setLastError(shafft::Status::ERR_MPI, SHAFFT_ERRSRC_MPI, freeRc);
          return SHAFFT_STATUS(SHAFFT_ERR_MPI);
        }
      }
      if (rc != SHAFFT_STATUS(SHAFFT_SUCCESS))
        return rc;

      if (globalOk) {
        bestD = dEffective;
        bestSubsize = std::move(subsize);
        bestOffset = std::move(offset);
        return SHAFFT_STATUS(SHAFFT_SUCCESS);
      }
      return SHAFFT_STATUS(SHAFFT_ERR_INVALID_DECOMP);
    };

    if (!autoMode) {
      int rc = tryD(*nda, true);
      if (rc != SHAFFT_STATUS(SHAFFT_SUCCESS))
        return rc;
    } else {
      if (wantMax) {
        for (int d = ndim - 1; d >= 1; --d) {
          int rc = tryD(d, false);
          if (rc == SHAFFT_STATUS(SHAFFT_SUCCESS))
            break;
          if (rc != SHAFFT_STATUS(SHAFFT_ERR_INVALID_DECOMP))
            return rc;
        }
      } else {
        for (int d = 1; d <= ndim - 1; ++d) {
          int rc = tryD(d, false);
          if (rc == SHAFFT_STATUS(SHAFFT_SUCCESS))
            break;
          if (rc != SHAFFT_STATUS(SHAFFT_ERR_INVALID_DECOMP))
            return rc;
        }
      }
      if (bestD < 0)
        return SHAFFT_STATUS(SHAFFT_ERR_INVALID_DECOMP);
    }

    *nda = bestD;
    std::copy(bestSubsize.begin(), bestSubsize.end(), subsizeOut);
    std::copy(bestOffset.begin(), bestOffset.end(), offsetOut);
    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

// Validate and use an explicitly specified Cartesian process grid.
// commDims must be fully specified (all > 0). memLimit == 0 means unlimited.
static int configurationCart(int ndim,
                             int size[],
                             int* subsizeOut,
                             int* offsetOut,
                             int* commDims,
                             int* commSize,
                             shafft::FFTType precision,
                             size_t memLimit,
                             MPI_Comm comm) noexcept {
  try {
    int worldSize = 0;
    SHAFFT_MPI_OR_FAIL(MPI_Comm_size(comm, &worldSize));

    // Validation lambda: checks grid constraints and computes local block
    auto validateAndBuild =
        [&](int d, const int* grid, std::vector<int>& subsize, std::vector<int>& offset) -> int {
      // Cross-axis constraint: commDims[i] <= min(sizes[i], sizes[i + nca])
      std::vector<int> caps;
      calcMinCaps(ndim, d, size, caps);
      for (int i = 0; i < d; ++i) {
        if (grid[i] < 1)
          return SHAFFT_STATUS(SHAFFT_ERR_INVALID_DECOMP);
        if (grid[i] > size[i])
          return SHAFFT_STATUS(SHAFFT_ERR_INVALID_DECOMP);
        if (grid[i] > caps[i])
          return SHAFFT_STATUS(SHAFFT_ERR_INVALID_DECOMP);
      }
      const int cartSz = shafft::detail::prodN(grid, d);
      if (cartSz > worldSize)
        return SHAFFT_STATUS(SHAFFT_ERR_INVALID_DECOMP);

      MPI_Comm cart = MPI_COMM_NULL;
      int rc = computeLocalBlockCart(ndim, size, d, grid, comm, subsize, offset, &cart);
      if (rc != SHAFFT_STATUS(SHAFFT_SUCCESS))
        return rc;

      bool localOk = (cart == MPI_COMM_NULL) || passesLimits(ndim, subsize, precision, memLimit);
      bool globalOk = false;
      rc = allreduceBoolAnd(localOk, &globalOk, comm);
      if (cart != MPI_COMM_NULL) {
        SHAFFT_MPI_OR_FAIL(MPI_Comm_free(&cart));
      }
      if (rc != SHAFFT_STATUS(SHAFFT_SUCCESS))
        return rc;

      return globalOk ? SHAFFT_STATUS(SHAFFT_SUCCESS) : SHAFFT_STATUS(SHAFFT_ERR_INVALID_DECOMP);
    };

    std::vector<int> bestSubsize(ndim), bestOffset(ndim);

    // Count leading distributed axes (those with commDims[i] > 1)
    int d = 0;
    while (d < ndim && commDims[d] > 1)
      ++d;

    // Handle single-rank case: d==0 means all commDims are 1, which is valid
    // for single rank (no distribution needed)
    if (d == 0) {
      // All commDims should be 1 for this to be valid
      bool allOnes = true;
      for (int i = 0; i < ndim; ++i) {
        if (commDims[i] != 1) {
          allOnes = false;
          break;
        }
      }
      if (!allOnes || worldSize != 1) {
        return SHAFFT_STATUS(SHAFFT_ERR_INVALID_DECOMP);
      }
      // Single rank: local block is the whole tensor
      for (int i = 0; i < ndim; ++i) {
        bestSubsize[i] = size[i];
        bestOffset[i] = 0;
      }
      *commSize = 1;
      std::copy(bestSubsize.begin(), bestSubsize.end(), subsizeOut);
      std::copy(bestOffset.begin(), bestOffset.end(), offsetOut);
      return SHAFFT_STATUS(SHAFFT_SUCCESS);
    }

    // Validate: distributed axes must be a prefix (no trailing >1 after first 1)
    if (d >= ndim)
      return SHAFFT_STATUS(SHAFFT_ERR_INVALID_DECOMP);
    for (int i = d; i < ndim; ++i) {
      if (commDims[i] > 1)
        return SHAFFT_STATUS(SHAFFT_ERR_INVALID_DECOMP);
      if (commDims[i] == 0)
        commDims[i] = 1; // normalize trailing zeros
    }

    std::vector<int> subsize, offset;
    int rc = validateAndBuild(d, commDims, subsize, offset);
    if (rc != SHAFFT_STATUS(SHAFFT_SUCCESS))
      return rc;

    *commSize = shafft::detail::prodN(commDims, d);
    std::copy(subsize.begin(), subsize.end(), subsizeOut);
    std::copy(offset.begin(), offset.end(), offsetOut);
    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

// ---- public API ------------------------------------------------------------

namespace shafft::detail {

int configurationND(int ndim,
                    int size[],
                    shafft::FFTType precision,
                    int* commDims,
                    int* nda,
                    int* subsizeOut,
                    int* offsetOut,
                    int* commSize,
                    shafft::DecompositionStrategy strategy,
                    size_t memLimit,
                    MPI_Comm comm) noexcept {
  // Unified configuration with cascading fallback: commDims -> nda -> strategy

  // Step 1: If commDims is fully specified (all > 0), try to use it
  if (!allZero(commDims, ndim)) {
    bool fullySpecified = true;
    for (int i = 0; i < ndim; ++i) {
      if (commDims[i] <= 0) {
        fullySpecified = false;
        break;
      }
    }
    if (fullySpecified) {
      int rc = configurationCart(
          ndim, size, subsizeOut, offsetOut, commDims, commSize, precision, memLimit, comm);
      if (rc == SHAFFT_STATUS(SHAFFT_SUCCESS)) {
        // Derive nda from commDims
        int d = 0;
        while (d < ndim && commDims[d] > 1)
          ++d;
        *nda = d;
        return SHAFFT_STATUS(SHAFFT_SUCCESS);
      }
      // Fallback: commDims failed, continue to step 2
    }
  }

  // Step 2: If nda > 0, try to achieve that exact nda
  if (*nda > 0) {
    int requestedNda = *nda;
    int rc = configurationNDA(ndim,
                              size,
                              nda,
                              subsizeOut,
                              offsetOut,
                              commDims,
                              precision,
                              memLimit,
                              comm,
                              false /*preferMinNd*/);
    if (rc == SHAFFT_STATUS(SHAFFT_SUCCESS) && *nda == requestedNda) {
      // Compute commSize from commDims
      *commSize = detail::prodN(commDims, *nda);
      return SHAFFT_STATUS(SHAFFT_SUCCESS);
    }
    // Fallback: requested nda failed, continue to step 3
  }

  // Step 3: Use strategy (MAXIMIZE_NDA or MINIMIZE_NDA)
  *nda = 0; // Auto-select mode
  bool preferMinNd = (strategy == shafft::DecompositionStrategy::MINIMIZE_NDA);
  int rc = configurationNDA(
      ndim, size, nda, subsizeOut, offsetOut, commDims, precision, memLimit, comm, preferMinNd);
  if (rc == SHAFFT_STATUS(SHAFFT_SUCCESS)) {
    // Compute commSize from commDims
    *commSize = detail::prodN(commDims, *nda);
    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }

  return rc;
}

} // namespace shafft::detail
