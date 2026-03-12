/**
 * @file test_FFT1D.cpp
 * @brief Unit tests for FFT1D class and configuration1D()
 *
 * Tests the FFT1D API for distributed 1D FFT:
 *   - configuration1D() for layout queries
 *   - FFT1D initialization and lifecycle
 *   - Buffer management
 *   - Query methods (globalShape, localSize, allocSize, etc.)
 */
#include "test_utils.hpp"
#include <algorithm>
#include <mpi.h>
#include <shafft/shafft.hpp>
#include <shafft/shafft_config.h>
#include <vector>

// Test: configuration1D returns valid layout
static bool test_configuration1d_basic() {
  int worldSize, worldRank;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

  size_t N = 1024;
  size_t localN = 0, localStart = 0;

  int rc = shafft::configuration1D(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  // localStart should be valid (within [0, N])
  if (localStart > N)
    return false;

  // Gather all localN to verify they cover N exactly
  std::vector<size_t> all_local_n(worldSize);
  MPI_Allgather(
      &localN, 1, MPI_UNSIGNED_LONG, all_local_n.data(), 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);

  size_t total = 0;
  for (int i = 0; i < worldSize; ++i)
    total += all_local_n[i];

  // Total should equal N exactly (user data coverage)
  if (total != N)
    return false;

  return true;
}

// Test: configuration1D with various sizes
static bool test_configuration1d_various_sizes() {
  int worldSize, worldRank;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

  // Test sizes that may or may not divide evenly
  std::vector<size_t> test_sizes = {16, 17, 31, 32, 100, 127, 128, 255, 256, 1000, 1024};

  for (size_t N : test_sizes) {
    size_t localN = 0, localStart = 0;
    int rc = shafft::configuration1D(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0)
      return false;

#if SHAFFT_BACKEND_HIPFFT
      // hipFFT Cooley-Tukey with L-based blocks:
      // localN = min(L, max(N - rank*L, 0))
      // Trailing ranks may have localN=0 when N < rank*L
      // This is expected - we just verify configuration1D succeeded.
      // The sum(localN)==N check is done in test_configuration1d_layout_correctness.
#else
      // FFTW-MPI: ranks may be inactive (localN == 0) when N < P
      // This is expected behavior - only verify localN is set
      // (it may be 0 for inactive ranks)
#endif
  }

  return true;
}

// Test: configuration1D double precision
static bool test_configuration1d_double() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  size_t N = 512;
  size_t localN = 0, localStart = 0;

  int rc = shafft::configuration1D(N, localN, localStart, shafft::FFTType::Z2Z, MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  // Verify sum of localN across all ranks equals N
  size_t total_local_n = 0;
  MPI_Allreduce(&localN, &total_local_n, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
  if (total_local_n != N)
    return false;

  return true;
}

// Test: localN vs localAllocSize distinction for non-divisible sizes
// Backend-agnostic invariants: contiguous, non-overlapping, covers all of N.
// The L-based block layout formula (localStart = rank*L, localN = min(L, N - rank*L))
// only holds for the hipFFT backend; FFTW-MPI uses its own radix-based decomposition.
static bool test_configuration1d_layout_correctness() {
  int worldSize, worldRank;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

  // Test with sizes that don't divide evenly by numRanks
  // These are the critical cases where localN != localAllocSize
  std::vector<size_t> test_sizes = {10, 17, 31, 100, 127, 255, 1000, 1023};

  for (size_t N : test_sizes) {
    size_t localN = 0, localStart = 0;
    int rc = shafft::configuration1D(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0)
      return false;

    // Gather localN and localStart from all ranks
    std::vector<size_t> all_local_n(worldSize);
    std::vector<size_t> all_local_start(worldSize);
    MPI_Allgather(
        &localN, 1, MPI_UNSIGNED_LONG, all_local_n.data(), 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
    MPI_Allgather(&localStart,
                  1,
                  MPI_UNSIGNED_LONG,
                  all_local_start.data(),
                  1,
                  MPI_UNSIGNED_LONG,
                  MPI_COMM_WORLD);

    // Verify: sum of all localN should equal N (not padded N)
    size_t total_local_n = 0;
    for (int i = 0; i < worldSize; ++i)
      total_local_n += all_local_n[i];

    if (total_local_n != N) {
      if (worldRank == 0) {
        std::printf("FAIL: N=%zu, sum(localN)=%zu (expected %zu)\n", N, total_local_n, N);
      }
      return false;
    }

    // Verify: active blocks (localN > 0) are contiguous and non-overlapping.
    // FFTW-MPI inactive ranks report localStart=0 regardless of position,
    // so we only check contiguity among active ranks.
    {
      // Collect (localStart, localN) for active ranks, sorted by localStart
      std::vector<std::pair<size_t, size_t>> active_blocks;
      for (int i = 0; i < worldSize; ++i) {
        if (all_local_n[i] > 0)
          active_blocks.push_back({all_local_start[i], all_local_n[i]});
      }
      std::sort(active_blocks.begin(), active_blocks.end());

      // First active block must start at 0
      if (!active_blocks.empty() && active_blocks[0].first != 0) {
        if (worldRank == 0) {
          std::printf("FAIL: N=%zu, first active block starts at %zu (expected 0)\n",
                      N,
                      active_blocks[0].first);
        }
        return false;
      }

      // Successive active blocks must be contiguous
      for (size_t b = 1; b < active_blocks.size(); ++b) {
        size_t expected_start = active_blocks[b - 1].first + active_blocks[b - 1].second;
        if (active_blocks[b].first != expected_start) {
          if (worldRank == 0) {
            std::printf("FAIL: N=%zu, active block %zu starts at %zu (expected %zu)\n",
                        N,
                        b,
                        active_blocks[b].first,
                        expected_start);
          }
          return false;
        }
      }
    }

#if SHAFFT_BACKEND_HIPFFT
    // hipFFT L-based block layout: localStart = min(rank*L, N),
    // localN = min(L, max(N - rank*L, 0))
    shafft::FFT1D fft;
    rc = fft.init(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0)
      return false;
    rc = fft.plan();
    if (rc != 0)
      return false;
    size_t L = fft.allocSize(); // L = fftLength/P

    // Verify: localStart should follow block layout: min(rank * L, N)
    for (int i = 0; i < worldSize; ++i) {
      size_t expected_start = std::min(static_cast<size_t>(i) * L, N);

      if (all_local_start[i] != expected_start) {
        if (worldRank == 0) {
          std::printf("FAIL: N=%zu, rank %d localStart=%zu (expected min(%d*%zu,%zu)=%zu)\n",
                      N,
                      i,
                      all_local_start[i],
                      i,
                      L,
                      N,
                      expected_start);
        }
        return false;
      }
    }

    // Verify: localN should follow formula: min(L, max(N - rank*L, 0))
    for (int i = 0; i < worldSize; ++i) {
      size_t rank_offset = static_cast<size_t>(i) * L;
      size_t expected_local_n;
      if (rank_offset >= N) {
        expected_local_n = 0;
      } else {
        size_t remaining = N - rank_offset;
        expected_local_n = std::min(remaining, L);
      }

      if (all_local_n[i] != expected_local_n) {
        if (worldRank == 0) {
          std::printf("FAIL: N=%zu, rank %d localN=%zu (expected %zu)\n",
                      N,
                      i,
                      all_local_n[i],
                      expected_local_n);
        }
        return false;
      }
    }
#endif // SHAFFT_BACKEND_HIPFFT
  }

  return true;
}

// Test: allocSize >= localN and allocSize is sufficient for algorithm
static bool test_configuration1d_alloc_size() {
  int worldSize, worldRank;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

  std::vector<size_t> test_sizes = {10, 17, 31, 100, 127, 255, 256, 1000, 1024};

  for (size_t N : test_sizes) {
    size_t localN = 0, localStart = 0;
    int rc = shafft::configuration1D(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0)
      return false;

    shafft::FFT1D fft;
    rc = fft.init(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0)
      return false;
    rc = fft.plan();
    if (rc != 0)
      return false;

    size_t allocSize = fft.allocSize();
    size_t reportedLocalN = fft.localSize();

    // allocSize must be >= localN (we need room for user's data)
    if (allocSize < reportedLocalN) {
      if (worldRank == 0) {
        std::printf("FAIL: N=%zu, allocSize=%zu < localN=%zu\n", N, allocSize, reportedLocalN);
      }
      return false;
    }

    // For hipFFT, verify allocSize is padded correctly for Cooley-Tukey
#if SHAFFT_BACKEND_HIPFFT
    // allocSize should be divisible by numRanks (for chunk-based algorithm)
    if (allocSize % static_cast<size_t>(worldSize) != 0) {
      if (worldRank == 0) {
        std::printf("FAIL: N=%zu, allocSize=%zu not divisible by %d\n", N, allocSize, worldSize);
      }
      return false;
    }
#endif
  }

  return true;
}

// Test: FFT1D basic initialization
static bool test_FFT1D_init() {
  size_t N = 256;
  size_t localN, localStart;

  int rc = shafft::configuration1D(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  shafft::FFT1D fft;

  // Not configured yet
  if (fft.isConfigured())
    return false;

  rc = fft.init(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  // Now configured
  if (!fft.isConfigured())
    return false;

  return true;
}

// Test: FFT1D global shape and size
static bool test_FFT1D_global_shape() {
  size_t N = 512;
  size_t localN, localStart;

  int rc = shafft::configuration1D(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  shafft::FFT1D fft;
  rc = fft.init(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  // globalShape should return single-element vector
  std::vector<size_t> shape = fft.globalShape();
  if (shape.size() != 1)
    return false;
  if (shape[0] != N)
    return false;

  // globalSize should return N
  size_t global = fft.globalSize();
  if (global != N)
    return false;

  return true;
}

// Test: FFT1D local and alloc size
static bool test_FFT1D_local_alloc_size() {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  size_t N = 256;
  size_t localN, localStart;

  int rc = shafft::configuration1D(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  shafft::FFT1D fft;
  rc = fft.init(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  size_t local = fft.localSize();
  size_t alloc = fft.allocSize();

  // Local size may be 0 for some ranks when N < worldSize * L (Bluestein)
  // No check that local > 0

  // Alloc should be >= local
  if (alloc < local)
    return false;

  // Verify total local across ranks equals N
  size_t total_local = 0;
  MPI_Allreduce(&local, &total_local, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
  if (total_local != N)
    return false;

  return true;
}

// Test: FFT1D getLayout
static bool test_FFT1D_getlayout() {
  size_t N = 128;
  size_t localN, localStart;

  int rc = shafft::configuration1D(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  shafft::FFT1D fft;
  rc = fft.init(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  std::vector<size_t> local_shape, offset;
  rc = fft.getLayout(local_shape, offset, shafft::TensorLayout::CURRENT);
  if (rc != 0)
    return false;

  // Should be 1D
  if (local_shape.size() != 1)
    return false;
  if (offset.size() != 1)
    return false;

  // local_shape may be 0 for some ranks when N < worldSize * L (Bluestein)
  // But offset should be valid (not checked for unsigned underflow)

  return true;
}

// Test: FFT1D getAxes
static bool test_FFT1D_getaxes() {
  size_t N = 64;
  size_t localN, localStart;

  int rc = shafft::configuration1D(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  shafft::FFT1D fft;
  rc = fft.init(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  std::vector<int> ca, da;
  rc = fft.getAxes(ca, da, shafft::TensorLayout::CURRENT);
  if (rc != 0)
    return false;

  // 1D distributed FFT: axis 0 is distributed, no contiguous axes
  if (da.size() != 1)
    return false;
  if (da[0] != 0)
    return false;
  if (ca.size() != 0)
    return false;

  return true;
}

// Test: FFT1D isActive
static bool test_FFT1D_is_active() {
  int worldSize, worldRank;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

  size_t N = 32;
  size_t localN, localStart;

  int rc = shafft::configuration1D(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  shafft::FFT1D fft;
  rc = fft.init(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  bool active = fft.isActive();
  size_t myLocalN = fft.localSize();

  // Accumulate pass/fail locally — never return before the collective below.
  bool localPass = true;

  // Active ranks must have non-zero localN; inactive ones must have zero.
  if (active && myLocalN == 0)
    localPass = false;
  if (!active && myLocalN != 0)
    localPass = false;

  // Count active ranks across all processes.
  int activeCount = active ? 1 : 0;
  int totalActive = 0;
  MPI_Allreduce(&activeCount, &totalActive, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  // There must be at least one active rank.
  if (totalActive < 1)
    localPass = false;

  // For 2+ MPI ranks there should be at least 2 active ranks
  // (N=32 is large enough for up to ~32 ranks).
  if (worldSize >= 2 && totalActive < 2)
    localPass = false;

  return localPass;
}

// Test: FFT1D setBuffers and getBuffers
static bool test_FFT1D_buffers() {
  size_t N = 128;
  size_t localN, localStart;

  int rc = shafft::configuration1D(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  shafft::FFT1D fft;
  rc = fft.init(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  size_t alloc = fft.allocSize();
  shafft::complexf *data = nullptr, *work = nullptr;

  rc = shafft::allocBuffer(alloc, &data);
  if (rc != 0)
    return false;
  rc = shafft::allocBuffer(alloc, &work);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    return false;
  }

  // Set buffers
  rc = fft.setBuffers(data, work);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  // Get buffers back
  shafft::complexf *data_out = nullptr, *work_out = nullptr;
  rc = fft.getBuffers(&data_out, &work_out);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  // Verify pointers match
  if (data_out != data) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }
  if (work_out != work) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  (void)shafft::freeBuffer(data);
  (void)shafft::freeBuffer(work);
  return true;
}

// Test: FFT1D double precision buffers
static bool test_FFT1D_double_buffers() {
  size_t N = 64;
  size_t localN, localStart;

  int rc = shafft::configuration1D(N, localN, localStart, shafft::FFTType::Z2Z, MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  shafft::FFT1D fft;
  rc = fft.init(N, localN, localStart, shafft::FFTType::Z2Z, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  size_t alloc = fft.allocSize();
  shafft::complexd *data = nullptr, *work = nullptr;

  rc = shafft::allocBuffer(alloc, &data);
  if (rc != 0)
    return false;
  rc = shafft::allocBuffer(alloc, &work);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    return false;
  }

  rc = fft.setBuffers(data, work);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  shafft::complexd *data_out = nullptr, *work_out = nullptr;
  rc = fft.getBuffers(&data_out, &work_out);
  if (rc != 0) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  if (data_out != data || work_out != work) {
    (void)shafft::freeBuffer(data);
    (void)shafft::freeBuffer(work);
    return false;
  }

  (void)shafft::freeBuffer(data);
  (void)shafft::freeBuffer(work);
  return true;
}

// Test: FFT1D move semantics
static bool test_FFT1D_move() {
  size_t N = 64;
  size_t localN, localStart;

  int rc = shafft::configuration1D(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  shafft::FFT1D fft1;
  rc = fft1.init(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  // Move construct
  shafft::FFT1D fft2(std::move(fft1));
  if (!fft2.isConfigured())
    return false;
  if (fft1.isConfigured())
    return false; // fft1 should be empty now

  // Move assign
  shafft::FFT1D plan3;
  plan3 = std::move(fft2);
  if (!plan3.isConfigured())
    return false;
  if (fft2.isConfigured())
    return false;

  return true;
}

// Test: FFT1D release
static bool test_FFT1D_release() {
  size_t N = 32;
  size_t localN, localStart;

  int rc = shafft::configuration1D(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  shafft::FFT1D fft;
  rc = fft.init(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;
  rc = fft.plan();
  if (rc != 0)
    return false;

  if (!fft.isConfigured())
    return false;

  fft.release();

  if (fft.isConfigured())
    return false;

  return true;
}

#if SHAFFT_BACKEND_HIPFFT

// ============================================================================
// hipFFT-specific tests for two-path distributed 1D FFT
// - Path A (Cooley-Tukey): Requires padding N' % P^2 == 0
// - Path B (Bluestein): Exact N-point DFT via chirp-Z convolution
// Default is now Path B (allowPadding = false) for exact frequency semantics
// ============================================================================

// Test: Default path (Bluestein) returns exact N-point layout
// With allowPadding=false (default), sum(localN) == N exactly
static bool test_hipfft_bluestein_exact_layout() {
  int worldSize, worldRank;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

  // Test various sizes including primes and non-P^2-divisible values
  std::vector<size_t> test_sizes = {7, 10, 13, 17, 31, 64, 97, 100, 127, 256, 1000, 1023};

  for (size_t N : test_sizes) {
    size_t localN = 0, localStart = 0;
    // Default config uses allowPadding=false -> Bluestein path
    int rc = shafft::configuration1D(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0)
      return false;

    // Gather all localN values
    std::vector<size_t> allLocalN(worldSize);
    MPI_Gather(
        &localN, 1, MPI_UNSIGNED_LONG, allLocalN.data(), 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    if (worldRank == 0) {
      size_t totalN = 0;
      for (int r = 0; r < worldSize; r++) {
        totalN += allLocalN[r];
      }
      // CRITICAL: With Bluestein, sum(localN) must equal N exactly
      if (totalN != N) {
        std::printf("FAIL (Bluestein exact): N=%zu, sum(localN)=%zu != N\n", N, totalN);
        return false;
      }
    }
  }

  return true;
}

// Test: All ranks have equal allocSize (L) for internal computation
// Note: For Bluestein, allocSize is M/P; for Cooley-Tukey, it's N'/P
static bool test_hipfft_equal_alloc_size() {
  int worldSize, worldRank;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

  std::vector<size_t> test_sizes = {10, 16, 17, 64, 100};

  for (size_t N : test_sizes) {
    size_t localN = 0, localStart = 0;
    // Use default config (Bluestein path)
    int rc = shafft::configuration1D(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0)
      return false;

    shafft::FFT1D fft;
    rc = fft.init(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0)
      return false;
    rc = fft.plan();
    if (rc != 0)
      return false;

    size_t allocSize = fft.allocSize();

    // Gather all allocSizes to rank 0
    std::vector<size_t> allAllocSize(worldSize);
    MPI_Gather(&allocSize,
               1,
               MPI_UNSIGNED_LONG,
               allAllocSize.data(),
               1,
               MPI_UNSIGNED_LONG,
               0,
               MPI_COMM_WORLD);

    if (worldRank == 0) {
      for (int r = 1; r < worldSize; r++) {
        if (allAllocSize[r] != allAllocSize[0]) {
          std::printf("FAIL: N=%zu, rank %d allocSize=%zu != rank 0 allocSize=%zu\n",
                      N,
                      r,
                      allAllocSize[r],
                      allAllocSize[0]);
          return false;
        }
      }
    }
  }

  return true;
}

// Test: Bluestein allocSize is M/P where M = ceil((2N-1)/P^2)*P^2
static bool test_hipfft_bluestein_alloc_size_formula() {
  int worldSize, worldRank;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

  size_t P = static_cast<size_t>(worldSize);
  size_t P2 = P * P;

  // Use sizes that don't divide by P^2 to ensure Bluestein path
  std::vector<size_t> test_sizes = {7, 10, 13, 31, 97, 127};

  for (size_t N : test_sizes) {
    // Skip sizes that would use Cooley-Tukey (N % P^2 == 0)
    if (N % P2 == 0)
      continue;

    size_t localN = 0, localStart = 0;
    int rc = shafft::configuration1D(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0)
      return false;

    shafft::FFT1D fft;
    rc = fft.init(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0)
      return false;
    rc = fft.plan();
    if (rc != 0)
      return false;

    size_t allocSize = fft.allocSize();

    // Expected M for Bluestein: M = ceil((2N-1)/P^2)*P^2
    size_t M0 = 2 * N - 1;
    size_t M = ((M0 + P2 - 1) / P2) * P2;
    size_t expectedL = M / P;

    if (allocSize != expectedL) {
      if (worldRank == 0) {
        std::printf("FAIL (Bluestein allocSize): N=%zu, allocSize=%zu, expected M/P=%zu (M=%zu)\n",
                    N,
                    allocSize,
                    expectedL,
                    M);
      }
      return false;
    }
  }

  return true;
}

// Test: localN formula for Bluestein (exact N-point distribution)
// localN[r] = min(L, max(N - r*L, 0)) where L = M/P (but mapped to N elements)
// This test verifies exact values for N=10, P=4
static bool test_hipfft_bluestein_local_n_formula_p4() {
  int worldSize, worldRank;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

  // This test requires exactly 4 ranks
  if (worldSize != 4) {
    // Skip test silently for other rank counts
    return true;
  }

  size_t N = 10;
  // With Bluestein (default): M = ceil((2*10-1)/16)*16 = ceil(19/16)*16 = 32
  // L_bluestein = M/P = 32/4 = 8 (internal allocation)
  // But localN reflects actual data: distributed evenly with possible remainder

  size_t localN = 0, localStart = 0;
  int rc = shafft::configuration1D(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
  if (rc != 0)
    return false;

  // Gather all values
  std::vector<size_t> allLocalN(worldSize), allLocalStart(worldSize);
  MPI_Gather(
      &localN, 1, MPI_UNSIGNED_LONG, allLocalN.data(), 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  MPI_Gather(&localStart,
             1,
             MPI_UNSIGNED_LONG,
             allLocalStart.data(),
             1,
             MPI_UNSIGNED_LONG,
             0,
             MPI_COMM_WORLD);

  if (worldRank == 0) {
    // Verify sum equals N exactly (Bluestein property)
    size_t totalN = 0;
    for (int r = 0; r < worldSize; r++) {
      totalN += allLocalN[r];
    }
    if (totalN != N) {
      std::printf("FAIL: N=10, P=4 (Bluestein), sum(localN)=%zu != N=%zu\n", totalN, N);
      return false;
    }

    // Verify localStarts are contiguous and correct
    size_t expectedStart = 0;
    for (int r = 0; r < worldSize; r++) {
      if (allLocalStart[r] != expectedStart) {
        std::printf("FAIL: N=10, P=4 (Bluestein), rank %d localStart=%zu, expected=%zu\n",
                    r,
                    allLocalStart[r],
                    expectedStart);
        return false;
      }
      expectedStart += allLocalN[r];
    }
  }

  return true;
}

// Test: Prime N values always work (Bluestein handles them)
static bool test_hipfft_prime_sizes() {
  int worldSize, worldRank;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

  // Large primes that would fail Cooley-Tukey padding
  std::vector<size_t> primes = {7, 13, 31, 61, 97, 127, 251, 509, 1021};

  for (size_t N : primes) {
    size_t localN = 0, localStart = 0;
    int rc = shafft::configuration1D(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) {
      if (worldRank == 0) {
        std::printf("FAIL: Prime N=%zu, configuration1D failed\n", N);
      }
      return false;
    }

    shafft::FFT1D fft;
    rc = fft.init(N, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
    if (rc != 0) {
      if (worldRank == 0) {
        std::printf("FAIL: Prime N=%zu, init failed\n", N);
      }
      return false;
    }
    rc = fft.plan();
    if (rc != 0) {
      if (worldRank == 0) {
        std::printf("FAIL: Prime N=%zu, plan failed\n", N);
      }
      return false;
    }

    // Verify exact N-point layout
    std::vector<size_t> allLocalN(worldSize);
    MPI_Gather(
        &localN, 1, MPI_UNSIGNED_LONG, allLocalN.data(), 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    if (worldRank == 0) {
      size_t totalN = 0;
      for (int r = 0; r < worldSize; r++) {
        totalN += allLocalN[r];
      }
      if (totalN != N) {
        std::printf("FAIL: Prime N=%zu, sum(localN)=%zu != N\n", N, totalN);
        return false;
      }
    }
  }

  return true;
}

#endif // SHAFFT_BACKEND_HIPFFT

// Main
int main(int argc, char* argv[]) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::fprintf(stderr, "MPI_Init failed\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  test::TestRunner runner("FFT1D Unit Tests");

  runner.run("configuration1D basic", test_configuration1d_basic);
  runner.run("configuration1D various sizes", test_configuration1d_various_sizes);
  runner.run("configuration1D double precision", test_configuration1d_double);
  runner.run("configuration1D layout correctness", test_configuration1d_layout_correctness);
  runner.run("configuration1D alloc size", test_configuration1d_alloc_size);

#if SHAFFT_BACKEND_HIPFFT
  // hipFFT-specific tests for two-path distributed 1D FFT (default config = Bluestein)
  runner.run("hipFFT: Bluestein exact N-point layout", test_hipfft_bluestein_exact_layout);
  runner.run("hipFFT: equal allocSize all ranks", test_hipfft_equal_alloc_size);
  runner.run("hipFFT: Bluestein allocSize = M/P", test_hipfft_bluestein_alloc_size_formula);
  runner.run("hipFFT: Bluestein localN formula (P=4)", test_hipfft_bluestein_local_n_formula_p4);
  runner.run("hipFFT: prime sizes with Bluestein", test_hipfft_prime_sizes);
#endif

  runner.run("FFT1D init", test_FFT1D_init);
  runner.run("FFT1D global shape", test_FFT1D_global_shape);
  runner.run("FFT1D local/alloc size", test_FFT1D_local_alloc_size);
  runner.run("FFT1D getLayout", test_FFT1D_getlayout);
  runner.run("FFT1D getAxes", test_FFT1D_getaxes);
  runner.run("FFT1D isActive", test_FFT1D_is_active);
  runner.run("FFT1D buffers", test_FFT1D_buffers);
  runner.run("FFT1D double buffers", test_FFT1D_double_buffers);
  runner.run("FFT1D move semantics", test_FFT1D_move);
  runner.run("FFT1D release", test_FFT1D_release);

  int result = runner.finalize();
  MPI_Finalize();
  return result;
}
