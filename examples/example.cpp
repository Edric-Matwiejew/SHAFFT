/// \example example.cpp
/// HIP-specific SHAFFT C++ example with explicit GPU memory management.
#include <shafft/shafft.hpp>

#include <cstdio>
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <mpi.h>
#include <vector>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  // MPI setup
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  [[maybe_unused]] int rc;
  constexpr int ndim = 3;
  constexpr int printCount = 4;
  std::vector<size_t> dims = {64, 64, 32};

  // Get configuration
  std::vector<int> commDims(ndim, 0);
  std::vector<size_t> subsize(ndim), offset(ndim);
  int nda = 0, commSize;
  rc = shafft::configurationND(dims,
                               shafft::FFTType::C2C,
                               commDims,
                               nda,
                               subsize,
                               offset,
                               commSize,
                               shafft::DecompositionStrategy::MINIMIZE_NDA,
                               0,
                               MPI_COMM_WORLD);

  // Create and plan FFT
  shafft::FFTND fft;
  rc = fft.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  rc = fft.plan();

  size_t allocSize = fft.allocSize();
  size_t localElems = subsize[0] * subsize[1] * subsize[2];

  // Allocate GPU buffers (HIP-specific)
  hipComplex *dData, *dWork;
  (void)hipMalloc(&dData, allocSize * sizeof(hipComplex));
  (void)hipMalloc(&dWork, allocSize * sizeof(hipComplex));

  // Initialize: delta function at origin (rank 0, index 0)
  std::vector<hipComplex> host(localElems, {0.0f, 0.0f});
  if (rank == 0 && localElems > 0)
    host[0] = {1.0f, 0.0f};

  (void)hipMemcpy(dData, host.data(), localElems * sizeof(hipComplex), hipMemcpyHostToDevice);
  (void)fft.setBuffers(dData, dWork);

  // Forward FFT
  (void)fft.execute(shafft::FFTDirection::FORWARD);
  (void)fft.normalize();

  // Retrieve spectrum
  (void)fft.getBuffers(&dData, &dWork);
  (void)hipMemcpy(host.data(), dData, localElems * sizeof(hipComplex), hipMemcpyDeviceToHost);

  if (rank == 0) {
    std::printf("Spectrum[0..%d] =", printCount - 1);
    for (int i = 0; i < printCount; ++i)
      std::printf(" (%g,%g)", host[i].x, host[i].y);
    std::printf("\n");
  }

  // Backward FFT
  (void)fft.setBuffers(dData, dWork);
  (void)fft.execute(shafft::FFTDirection::BACKWARD);
  (void)fft.normalize();

  // Retrieve result
  (void)fft.getBuffers(&dData, &dWork);
  (void)hipMemcpy(host.data(), dData, localElems * sizeof(hipComplex), hipMemcpyDeviceToHost);

  if (rank == 0) {
    std::printf("Result[0..%d]   =", printCount - 1);
    for (int i = 0; i < printCount; ++i)
      std::printf(" (%g,%g)", host[i].x, host[i].y);
    std::printf("\n");
  }

  // Cleanup
  (void)hipFree(dData);
  (void)hipFree(dWork);
  fft.release();

  MPI_Finalize();
  return 0;
}
