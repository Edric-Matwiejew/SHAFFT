#include <shafft/shafft_error.hpp>
#include <shafft/shafft_types.hpp>

#include "comm_dims.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>  // std::multiplies
#include <limits>
#include <mpi.h>
#include <numeric>
#include <vector>

// ---- helpers ---------------------------------------------------------------

static bool within_int_limit_exact(int ndim, const int dims[]) noexcept {
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

static bool within_mem_limit_guarded(int ndim, const int size_per_rank[], shafft::FFTType precision,
                                     size_t mem_limit) noexcept {
  // mem_limit <= 0 => ignore memory constraint
  if (mem_limit == 0)
    return true;
  if (static_cast<long long>(mem_limit) < 0)
    return true;

  size_t type_size = 0;
  switch (precision) {
    case shafft::FFTType::C2C:
      type_size = 2 * sizeof(float);
      break;
    case shafft::FFTType::Z2Z:
      type_size = 2 * sizeof(double);
      break;
    default:
      return false;
  }

  // Each rank holds both data and work buffers concurrently.
  const size_t buffer_count = 2;

  __uint128_t elems = 1;
  for (int i = 0; i < ndim; ++i) {
    if (size_per_rank[i] <= 0)
      return false;
    elems *= static_cast<uint64_t>(size_per_rank[i]);
    if (elems > ((__uint128_t)mem_limit) / (buffer_count * type_size))
      return false;
  }
  return (elems * buffer_count * type_size) <= mem_limit;
}

static void compute_block_1d(int N, int P, int p, int* n, int* s) {
  const int q = N / P;
  const int r = N % P;
  *n = q + (p < r ? 1 : 0);
  *s = (p < r) ? p * (q + 1) : r * (q + 1) + (p - r) * q;
}

static int allreduce_bool_and(bool local_ok, bool* global_ok, MPI_Comm comm) noexcept {
  if (!global_ok)
    return SHAFFT_STATUS(SHAFFT_ERR_NULLPTR);

  *global_ok = false;

  bool reduced = false;

  SHAFFT_MPI_OR_FAIL(MPI_Allreduce(&local_ok, &reduced, 1, MPI_C_BOOL, MPI_LAND, comm));

  *global_ok = reduced;
  return SHAFFT_STATUS(SHAFFT_SUCCESS);
}

static int prod_prefix(const int* a, int n) {
  int p = 1;
  for (int i = 0; i < n; ++i)
    p *= a[i];
  return p;
}

static bool all_zero(const int* a, int n) {
  for (int i = 0; i < n; ++i)
    if (a[i] != 0)
      return false;
  return true;
}

// Cross-axis constraint: COMM_DIMS[i] <= min(sizes[i], sizes[i + nca]) for i < d
// where nca = ndim - d. During exchanges, distributed axis i will partition
// contiguous axis (i + nca), so the communicator size must not exceed that axis's size.
static void calc_min_caps(int ndim, int d, const int* sizes, std::vector<int>& caps) {
  caps.assign(d, 1);
  int nca = ndim - d;
  for (int i = 0; i < d; ++i)
    caps[i] = std::min(sizes[i], sizes[i + nca]);
}

// Largest divisor of n that is <= limit (>=2 if possible, otherwise 1)
static int largest_divisor_leq(int n, int limit) {
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
        return best;  // can't do better than limit
    }
  }
  // If none of the proper factors fit, check n itself
  if (n <= up)
    best = std::max(best, n);
  return best;
}

// Pack candidate COMM_DIMS to a left prefix with "spill", respecting caps/sizes.
// We treat each >1 entry as a multiplicative factor that can be split across
// axes using integer divisors. We keep the last tensor axis contiguous.
// Returns d' = number of leading >1 entries after packing (0..ndim-1).
static int pack_prefix_with_spill(int ndim, int* grid, const int* size,
                                  const int* caps /*nullable*/) {
  // Collect multiplicative factors in order
  std::vector<int> factors;
  factors.reserve(ndim);
  for (int i = 0; i < ndim; ++i)
    if (grid[i] > 1)
      factors.push_back(grid[i]);

  // Reset grid to 1s; we'll rebuild the prefix
  for (int i = 0; i < ndim; ++i)
    grid[i] = 1;

  const int max_axes = std::max(0, ndim - 1);  // keep last axis contiguous
  int axis = 0;

  // Remaining capacity per axis (start at <= cap/size)
  auto axis_cap = [&](int i) -> int {
    int c = size[i];
    if (caps)
      c = std::min(c, caps[i]);
    return std::max(1, c);
  };

  std::vector<int> cap_left(max_axes, 1);
  for (int i = 0; i < max_axes; ++i)
    cap_left[i] = axis_cap(i);

  for (int f : factors) {
    int rem = f;
    while (rem > 1 && axis < max_axes) {
      // how much more we can multiply on this axis
      int room = cap_left[axis] / grid[axis];
      if (room < 2) {
        ++axis;
        continue;
      }
      int used = largest_divisor_leq(rem, room);
      if (used <= 1) {
        ++axis;
        continue;
      }
      grid[axis] *= used;
      rem /= used;
      // If we filled this axis up, move on; else we may continue placing here
      if (grid[axis] == cap_left[axis])
        ++axis;
    }
    // If rem > 1 here, we couldn't place all ranks; they'll become inactive.
  }

  // Count leading >1 (d')
  int d_prime = 0;
  while (d_prime < max_axes && grid[d_prime] > 1)
    ++d_prime;

  // Ensure remaining entries are 1 (including last axis)
  for (int i = d_prime; i < ndim; ++i)
    grid[i] = 1;
  return d_prime;
}

// Build the local block for a given grid. Creates a subcomm of size product(grid).

static bool passes_limits(int ndim, const std::vector<int>& subsize, shafft::FFTType precision,
                          size_t mem_limit, bool require_mem) {
  if (!within_int_limit_exact(ndim, subsize.data()))
    return false;
  if (require_mem && !within_mem_limit_guarded(ndim, subsize.data(), precision, mem_limit))
    return false;
  return true;
}

static int compute_local_block_cart(int ndim, const int* global_size, int d, const int* grid,
                                    MPI_Comm COMM, std::vector<int>& subsize,
                                    std::vector<int>& offset, MPI_Comm* cart_out) noexcept {
  subsize.assign(ndim, 0);
  offset.assign(ndim, 0);
  if (!cart_out)
    return SHAFFT_STATUS(SHAFFT_ERR_NULLPTR);
  *cart_out = MPI_COMM_NULL;

  if (d <= 0) {
    for (int i = 0; i < ndim; ++i) {
      subsize[i] = global_size[i];
      offset[i] = 0;
    }
    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }

  int cart_size = 1;
  for (int i = 0; i < d; ++i)
    cart_size *= grid[i];

  int world_rank = 0;
  SHAFFT_MPI_OR_FAIL(MPI_Comm_rank(COMM, &world_rank));

  MPI_Comm sub = MPI_COMM_NULL;
  const int color = (world_rank < cart_size) ? 1 : MPI_UNDEFINED;
  SHAFFT_MPI_OR_FAIL(MPI_Comm_split(COMM, color, world_rank, &sub));

  // Ranks not in subcomm: no work; success with cart_out = MPI_COMM_NULL
  if (sub == MPI_COMM_NULL)
    return SHAFFT_STATUS(SHAFFT_SUCCESS);

  std::vector<int> periods(d, 0);
  SHAFFT_MPI_OR_FAIL(
      MPI_Cart_create(sub, d, const_cast<int*>(grid), periods.data(), /*reorder=*/0, cart_out));
  // Free the intermediate subcomm
  SHAFFT_MPI_OR_FAIL(MPI_Comm_free(&sub));

  int cart_rank = -1;
  SHAFFT_MPI_OR_FAIL(MPI_Comm_rank(*cart_out, &cart_rank));

  std::vector<int> coords(d, 0);
  SHAFFT_MPI_OR_FAIL(MPI_Cart_coords(*cart_out, cart_rank, d, coords.data()));

  for (int i = 0; i < d; ++i) {
    int n_i, s_i;
    compute_block_1d(global_size[i], grid[i], coords[i], &n_i, &s_i);
    subsize[i] = n_i;
    offset[i] = s_i;
  }
  for (int i = d; i < ndim; ++i) {
    subsize[i] = global_size[i];
    offset[i] = 0;
  }

  return SHAFFT_STATUS(SHAFFT_SUCCESS);
}

// ---- public API ------------------------------------------------------------

namespace _shafft {

int configurationNDA(int ndim, int size[], int* nda, int* subsize_out, int* offset_out,
                     int* COMM_DIMS, shafft::FFTType precision, size_t mem_limit,
                     MPI_Comm COMM) noexcept {
  try {
    int world_size = 0;
    SHAFFT_MPI_OR_FAIL(MPI_Comm_size(COMM, &world_size));

    const bool auto_mode = (*nda == 0);
    const bool require_mem = (mem_limit > 0);
    const bool want_max = (mem_limit >= 0);  // mem==0 or >0 -> maximise; mem<0 -> minimise

    int best_d = -1;
    std::vector<int> best_subsize(ndim), best_offset(ndim);

    auto try_d = [&](int d, bool strict) -> bool {
      if (d < 1 || d > ndim - 1)
        return false;
      if (create_comm_dims(ndim, world_size, d, size, COMM_DIMS) != 0)
        return false;

      // In explicit/strict mode, all requested distributed axes must be >1.
      // If MPI_Dims_create (via create_comm_dims) could not populate every axis,
      // reject the decomposition instead of silently reducing nda.
      if (strict) {
        int gt1 = 0;
        for (int i = 0; i < d; ++i)
          if (COMM_DIMS[i] > 1)
            ++gt1;
        if (gt1 != d)
          return false;
      }

      std::vector<int> caps;
      calc_min_caps(ndim, d, size, caps);

      int d_effective = d;
      if (!strict) {
        // In auto mode, allow packing to consolidate the grid for efficiency
        d_effective = pack_prefix_with_spill(ndim, COMM_DIMS, size, caps.data());
        if (d_effective == 0 && world_size > 1)
          return false;

        // Recompute caps for the effective d after packing
        if (d_effective != d) {
          calc_min_caps(ndim, d_effective, size, caps);
          // Verify the packed grid satisfies the cross-axis constraint
          for (int i = 0; i < d_effective; ++i) {
            if (COMM_DIMS[i] > caps[i])
              return false;
          }
        }
      }
      // In strict mode, use the grid from create_comm_dims directly (no packing)

      std::vector<int> subsize, offset;
      MPI_Comm cart = MPI_COMM_NULL;
      int rc = compute_local_block_cart(ndim, size, d_effective, COMM_DIMS, COMM, subsize, offset,
                                        &cart);
      if (rc != SHAFFT_STATUS(SHAFFT_SUCCESS))
        return false;

      bool local_ok = (cart == MPI_COMM_NULL) ||
                      passes_limits(ndim, subsize, precision, mem_limit, require_mem);
      bool global_ok = false;
      rc = allreduce_bool_and(local_ok, &global_ok, COMM);
      if (cart != MPI_COMM_NULL) {
        SHAFFT_MPI_OR_FAIL(MPI_Comm_free(&cart));
      }
      if (rc != SHAFFT_STATUS(SHAFFT_SUCCESS))
        return false;

      if (global_ok) {
        best_d = d_effective;
        best_subsize = std::move(subsize);
        best_offset = std::move(offset);
        return true;
      }
      return false;
    };

    if (!auto_mode) {
      if (!try_d(*nda, true))
        return SHAFFT_STATUS(SHAFFT_ERR_INVALID_DECOMP);
    } else {
      if (want_max) {
        for (int d = ndim - 1; d >= 1; --d)
          if (try_d(d, false))
            break;
      } else {
        for (int d = 1; d <= ndim - 1; ++d)
          if (try_d(d, false))
            break;
      }
      if (best_d < 0)
        return SHAFFT_STATUS(SHAFFT_ERR_INVALID_DECOMP);
    }

    *nda = best_d;
    std::copy(best_subsize.begin(), best_subsize.end(), subsize_out);
    std::copy(best_offset.begin(), best_offset.end(), offset_out);
    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

int configurationCart(int ndim, int size[], int* subsize_out, int* offset_out, int* COMM_DIMS,
                      int* COMM_SIZE, shafft::FFTType precision, size_t mem_limit,
                      MPI_Comm COMM) noexcept {
  try {
    int world_size = 0;
    SHAFFT_MPI_OR_FAIL(MPI_Comm_size(COMM, &world_size));

    const bool require_mem = (mem_limit > 0);
    const bool want_max = (mem_limit >= 0);

    auto validate_and_build = [&](int d, const int* grid, std::vector<int>& subsize,
                                  std::vector<int>& offset) -> int {
      std::vector<int> caps;
      calc_min_caps(ndim, d, size, caps);
      for (int i = 0; i < d; ++i) {
        if (grid[i] < 1)
          return SHAFFT_STATUS(SHAFFT_ERR_INVALID_DECOMP);
        if (grid[i] > size[i])
          return SHAFFT_STATUS(SHAFFT_ERR_INVALID_DECOMP);
        if (grid[i] > caps[i])
          return SHAFFT_STATUS(SHAFFT_ERR_INVALID_DECOMP);
      }
      const int cart_sz = prod_prefix(grid, d);
      if (cart_sz > world_size)
        return SHAFFT_STATUS(SHAFFT_ERR_INVALID_DECOMP);

      MPI_Comm cart = MPI_COMM_NULL;
      int rc = compute_local_block_cart(ndim, size, d, grid, COMM, subsize, offset, &cart);
      if (rc != SHAFFT_STATUS(SHAFFT_SUCCESS))
        return rc;

      bool local_ok = (cart == MPI_COMM_NULL) ||
                      passes_limits(ndim, subsize, precision, mem_limit, require_mem);
      bool global_ok = false;
      rc = allreduce_bool_and(local_ok, &global_ok, COMM);
      if (cart != MPI_COMM_NULL) {
        SHAFFT_MPI_OR_FAIL(MPI_Comm_free(&cart));
      }
      if (rc != SHAFFT_STATUS(SHAFFT_SUCCESS))
        return rc;

      return global_ok ? SHAFFT_STATUS(SHAFFT_SUCCESS) : SHAFFT_STATUS(SHAFFT_ERR_INVALID_DECOMP);
    };

    int best_d = -1;
    int best_comm_size = 0;
    std::vector<int> best_subsize(ndim), best_offset(ndim);

    if (!all_zero(COMM_DIMS, ndim)) {
      int d = 0;
      while (d < ndim && COMM_DIMS[d] > 1)
        ++d;

      // Handle single-rank case: d==0 means all COMM_DIMS are 1, which is valid
      // for single rank (no distribution needed)
      if (d == 0) {
        // All COMM_DIMS should be 1 for this to be valid
        bool all_ones = true;
        for (int i = 0; i < ndim; ++i) {
          if (COMM_DIMS[i] != 1) {
            all_ones = false;
            break;
          }
        }
        if (!all_ones || world_size != 1) {
          return SHAFFT_STATUS(SHAFFT_ERR_INVALID_DECOMP);
        }
        // Single rank: local block is the whole tensor
        for (int i = 0; i < ndim; ++i) {
          best_subsize[i] = size[i];
          best_offset[i] = 0;
        }
        best_d = 0;
        best_comm_size = 1;
      } else {
        if (d >= ndim)
          return SHAFFT_STATUS(SHAFFT_ERR_INVALID_DECOMP);
        for (int i = d; i < ndim; ++i) {
          if (COMM_DIMS[i] > 1)
            return SHAFFT_STATUS(SHAFFT_ERR_INVALID_DECOMP);
          if (COMM_DIMS[i] == 0)
            COMM_DIMS[i] = 1;  // normalize trailing zeros
        }

        std::vector<int> subsize, offset;
        int rc = validate_and_build(d, COMM_DIMS, subsize, offset);
        if (rc != SHAFFT_STATUS(SHAFFT_SUCCESS))
          return rc;

        best_d = d;
        best_comm_size = prod_prefix(COMM_DIMS, d);
        best_subsize = std::move(subsize);
        best_offset = std::move(offset);
      }
    } else {
      auto try_d = [&](int d) -> int {
        if (create_comm_dims(ndim, world_size, d, size, COMM_DIMS) != 0)
          return SHAFFT_STATUS(SHAFFT_ERR_INVALID_DECOMP);

        std::vector<int> caps;
        calc_min_caps(ndim, d, size, caps);
        int d_packed = pack_prefix_with_spill(ndim, COMM_DIMS, size, caps.data());
        if (d_packed == 0 && world_size > 1)
          return SHAFFT_STATUS(SHAFFT_ERR_INVALID_DECOMP);

        // Recompute caps for the effective d after packing and validate
        if (d_packed != d) {
          calc_min_caps(ndim, d_packed, size, caps);
          for (int i = 0; i < d_packed; ++i) {
            if (COMM_DIMS[i] > caps[i])
              return SHAFFT_STATUS(SHAFFT_ERR_INVALID_DECOMP);
          }
        }

        std::vector<int> subsize, offset;
        int rc = validate_and_build(d_packed, COMM_DIMS, subsize, offset);
        if (rc != SHAFFT_STATUS(SHAFFT_SUCCESS))
          return rc;

        best_d = d_packed;
        best_comm_size = prod_prefix(COMM_DIMS, d_packed);
        best_subsize = std::move(subsize);
        best_offset = std::move(offset);
        return SHAFFT_STATUS(SHAFFT_SUCCESS);
      };

      int rc = SHAFFT_STATUS(SHAFFT_ERR_INVALID_DECOMP);
      if (want_max) {
        for (int d = ndim - 1; d >= 1; --d) {
          rc = try_d(d);
          if (rc == SHAFFT_STATUS(SHAFFT_SUCCESS))
            break;
        }
      } else {
        for (int d = 1; d <= ndim - 1; ++d) {
          rc = try_d(d);
          if (rc == SHAFFT_STATUS(SHAFFT_SUCCESS))
            break;
        }
      }
      if (best_d < 0)
        return rc;
    }

    *COMM_SIZE = best_comm_size;
    std::copy(best_subsize.begin(), best_subsize.end(), subsize_out);
    std::copy(best_offset.begin(), best_offset.end(), offset_out);
    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

}  // namespace _shafft
