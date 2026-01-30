// comm_dims.cpp
#include "comm_dims.h"

#include <mpi.h>
#include <functional>
#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

// ----------------------- Local helpers (file-private) -----------------------

static int product_n(const int* a, int n) {
  return std::accumulate(a, a + n, 1, std::multiplies<int>());
}

// Compute per-axis caps implied by the redistribution pattern:
// subcomm (nda - i - 1) <= min(size[i], size[ndim - i - 1])
static void calculate_min_shared_axes_size(int nda, int ndim,
                                           const int* sizes,
                                           int* min_sizes) {
  std::copy(sizes, sizes + nda, min_sizes);
  for (int i = 0; i < nda; ++i) {
    min_sizes[nda - i - 1] =
        std::min(min_sizes[nda - i - 1], sizes[ndim - i - 1]);
  }
}

// ------------------------------ Public function ------------------------------

int create_comm_dims(int ndim, int COMM_SIZE, int nda,
                     const int* sizes, int* COMM_DIMS)
{
  // Initialise: all dimensions set to 1
  std::fill(COMM_DIMS, COMM_DIMS + ndim, 1);

  if (nda <= 0 || nda > ndim) return 1;

  if (nda == 1) {
    // Single distributed axis: cap by axis size
    COMM_DIMS[0] = std::min(COMM_SIZE, sizes[0]);
    return 0;
  }

  // Compute per-axis caps implied by redistribution
  std::vector<int> min_sizes(nda);
  calculate_min_shared_axes_size(nda, ndim, sizes, min_sizes.data());

  // Use MPI_Dims_create for balanced factor distribution
  std::vector<int> mpi_dims(nda, 0);
  MPI_Dims_create(COMM_SIZE, nda, mpi_dims.data());

  // Apply caps: each COMM_DIM[i] = min(mpi_dims[i], min_sizes[i])
  for (int i = 0; i < nda; ++i) {
    COMM_DIMS[i] = std::min(mpi_dims[i], min_sizes[i]);
    if (COMM_DIMS[i] < 1) COMM_DIMS[i] = 1;
  }

  // Success criteria: non-zero product
  const int prod = product_n(COMM_DIMS, nda);
  if (prod <= 0) return 2;

  return 0;
}

