/**
 * @file test_fft_highdim.cpp
 * @brief Validate high-dimensional FFT correctness (hipFFT decomposition) against a naive DFT.
 *
 * Single-rank, no MPI redistribution (nda=0). Uses small tensor sizes to keep
 * the O(N^2) reference affordable while exercising multi-subplan (>3D) paths.
 */

#include <cmath>
#include <complex>
#include <cstdio>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <shafft/shafft.hpp>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

using cplx = std::complex<double>;

// Naive full DFT of an N-dimensional tensor in row-major order.
// Parallelized with OpenMP for speed.
static std::vector<cplx> dft_full(const std::vector<cplx>& in, const std::vector<size_t>& dims) {
  const int ndim = static_cast<int>(dims.size());
  // strides
  std::vector<size_t> stride(ndim, 1);
  for (int i = ndim - 2; i >= 0; --i)
    stride[i] = stride[i + 1] * dims[i + 1];
  const size_t total = in.size();
  std::vector<cplx> out(total, cplx{0.0, 0.0});
  const double two_pi = 2.0 * std::acos(-1.0);

#pragma omp parallel for schedule(dynamic)
  for (size_t fk = 0; fk < total; ++fk) {
    // Thread-local index arrays
    std::vector<size_t> k(ndim), x(ndim);

    size_t tmp = fk;
    for (int i = 0; i < ndim; ++i) {
      k[i] = tmp / stride[i];
      tmp %= stride[i];
    }

    cplx sum{0.0, 0.0};
    for (size_t fx = 0; fx < total; ++fx) {
      tmp = fx;
      for (int i = 0; i < ndim; ++i) {
        x[i] = tmp / stride[i];
        tmp %= stride[i];
      }
      double phase = 0.0;
      for (int i = 0; i < ndim; ++i)
        phase += static_cast<double>(k[i] * x[i]) / static_cast<double>(dims[i]);
      const double ang = -two_pi * phase;
      const double c = std::cos(ang), s = std::sin(ang);
      sum += in[fx] * cplx{c, s};
    }
    out[fk] = sum;
  }
  return out;
}

static std::vector<cplx> make_input(size_t n) {
  std::vector<cplx> v(n);
  for (size_t i = 0; i < n; ++i)
    v[i] = cplx(std::sin(0.01 * i), std::cos(0.02 * i));
  return v;
}

struct Case {
  std::vector<size_t> dims;
};

static bool run_case(const Case& tc, double tol = 1e-10) {
  const size_t total = std::accumulate(
      tc.dims.begin(), tc.dims.end(), size_t{1}, [](size_t a, size_t b) { return a * b; });

  // For nda=0 (no distribution), commDims should be all 1s
  std::vector<int> commDims(tc.dims.size(), 1);

  shafft::ConfigND cfg(tc.dims, shafft::FFTType::Z2Z, commDims);

  shafft::FFTND fft;
  int rc = fft.init(cfg.cStruct());
  if (rc != 0) {
    std::cerr << "fft.init rc=" << rc << "\n";
    return false;
  }
  rc = fft.plan();
  if (rc != 0) {
    std::cerr << "fft.plan rc=" << rc << "\n";
    return false;
  }

  auto host_in = make_input(total);
  shafft::complexd *d_data = nullptr, *d_work = nullptr;
  rc = shafft::allocBuffer(total, &d_data);
  if (rc != 0)
    return false;
  rc = shafft::allocBuffer(total, &d_work);
  if (rc != 0) {
    (void)shafft::freeBuffer(d_data);
    return false;
  }
  rc = shafft::copyToBuffer(
      d_data, reinterpret_cast<const shafft::complexd*>(host_in.data()), total);
  if (rc != 0) {
    (void)shafft::freeBuffer(d_data);
    (void)shafft::freeBuffer(d_work);
    return false;
  }
  rc = fft.setBuffers(d_data, d_work);
  if (rc != 0) {
    (void)shafft::freeBuffer(d_data);
    (void)shafft::freeBuffer(d_work);
    return false;
  }

  rc = fft.execute(shafft::FFTDirection::FORWARD);
  if (rc != 0) {
    std::cerr << "execute rc=" << rc << "\n";
    (void)shafft::freeBuffer(d_data);
    (void)shafft::freeBuffer(d_work);
    return false;
  }

  shafft::complexd *d_out, *d_unused;
  rc = fft.getBuffers(&d_out, &d_unused);
  if (rc != 0) {
    (void)shafft::freeBuffer(d_data);
    (void)shafft::freeBuffer(d_work);
    return false;
  }

  std::vector<cplx> host_out(total);
  rc = shafft::copyFromBuffer(reinterpret_cast<shafft::complexd*>(host_out.data()), d_out, total);
  if (rc != 0) {
    (void)shafft::freeBuffer(d_data);
    (void)shafft::freeBuffer(d_work);
    return false;
  }

  auto ref = dft_full(host_in, tc.dims);

  double max_err = 0.0;
  for (size_t i = 0; i < total; ++i) {
    double er = std::abs(host_out[i].real() - ref[i].real());
    double ei = std::abs(host_out[i].imag() - ref[i].imag());
    if (er > max_err)
      max_err = er;
    if (ei > max_err)
      max_err = ei;
  }

  (void)shafft::freeBuffer(d_data);
  (void)shafft::freeBuffer(d_work);

  if (max_err > tol) {
    std::cerr << "FAIL dims=";
    for (auto d : tc.dims)
      std::cerr << d << " ";
    std::cerr << " max_err=" << max_err << "\n";
    return false;
  }
  return true;
}

int main(int argc, char** argv) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::fprintf(stderr, "MPI_Init failed\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // This test is single-rank only (nda=0, commDims all 1s)
  int worldSize = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  if (worldSize > 1) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
      std::cout << "test_fft_highdim: SKIPPED (single-rank test, run with mpirun -n 1)\n";
    }
    MPI_Finalize();
    return 0;
  }

  std::vector<Case> cases = {
      // 5D through 12D with asymmetric dimension sizes (mostly small to limit superbatches)
      {{3u, 2u, 4u, 3u, 2u}},                             // 5D: 144 elements
      {{2u, 3u, 2u, 3u, 2u, 3u}},                         // 6D: 216 elements -> 3+3 subplans
      {{3u, 2u, 3u, 2u, 3u, 2u, 3u}},                     // 7D: 648 elements -> 3+3+1 subplans
      {{2u, 3u, 2u, 3u, 2u, 3u, 2u, 3u}},                 // 8D: 1296 elements -> 3+3+2 subplans
      {{3u, 2u, 3u, 2u, 3u, 2u, 3u, 2u, 3u}},             // 9D: 3888 elements -> 3+3+3 subplans
      {{2u, 3u, 2u, 3u, 2u, 3u, 2u, 3u, 2u, 3u}},         // 10D: 7776 elements -> 3+3+3+1 subplans
      {{2u, 3u, 2u, 3u, 2u, 3u, 2u, 3u, 2u, 3u, 2u}},     // 11D: 15552 elements -> 3+3+3+2 subplans
      {{2u, 2u, 2u, 3u, 2u, 2u, 2u, 3u, 2u, 2u, 2u, 3u}}, // 12D: 13824 elements -> 3+3+3+3 subplans
                                                          // (mostly 2s)
  };

  int failed = 0, passed = 0;
  for (const auto& tc : cases) {
    // Strict tolerance - errors above this indicate a bug
    double tol = 1e-10;
    if (run_case(tc, tol))
      passed++;
    else
      failed++;
  }

  if (failed == 0) {
    std::cout << "ALL PASSED (" << passed << ")\n";
    MPI_Finalize();
    return 0;
  }
  std::cout << failed << " FAILED, " << passed << " passed\n";
  MPI_Finalize();
  return 1;
}
