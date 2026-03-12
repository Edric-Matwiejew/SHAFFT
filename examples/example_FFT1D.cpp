/// \example example_FFT1D.cpp
/// Backend-portable SHAFFT C++ example using FFT1D.
#include <shafft/shafft.hpp>

#include <cstdio>
#include <mpi.h>
#include <vector>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  // MPI setup
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  [[maybe_unused]] int rc;
  constexpr size_t globalN = 256;
  constexpr int printCount = 4;

  // Get configuration
  size_t localN, localStart;
  rc = shafft::configuration1D(globalN, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);

  // Create and plan FFT
  shafft::FFT1D fft;
  rc = fft.init(globalN, localN, localStart, shafft::FFTType::C2C, MPI_COMM_WORLD);
  rc = fft.plan();

  size_t allocSize = fft.allocSize();
  size_t localSize = fft.localSize();

  // Allocate buffers
  shafft::complexf *data = nullptr, *work = nullptr;
  rc = shafft::allocBuffer(allocSize, &data);
  rc = shafft::allocBuffer(allocSize, &work);

  // Initialize: delta function at origin (index 0)
  std::vector<shafft::complexf> host(allocSize, {0.0f, 0.0f});
  if (localStart == 0 && localSize > 0)
    host[0] = {1.0f, 0.0f};

  rc = shafft::copyToBuffer(data, host.data(), allocSize);
  rc = fft.setBuffers(data, work);

  // Forward FFT
  rc = fft.execute(shafft::FFTDirection::FORWARD);
  rc = fft.normalize();

  // Retrieve spectrum
  std::vector<shafft::complexf> spectrum(allocSize);
  rc = shafft::copyFromBuffer(spectrum.data(), work, allocSize);

  if (rank == 0) {
    std::printf("Spectrum[0..%d] =", printCount - 1);
    for (int i = 0; i < printCount; ++i)
      std::printf(" (%g,%g)", spectrum[i].real(), spectrum[i].imag());
    std::printf("\n");
  }

  // Backward FFT
  rc = fft.setBuffers(work, data);
  rc = fft.execute(shafft::FFTDirection::BACKWARD);
  rc = fft.normalize();

  // Retrieve result
  shafft::complexf *curData, *curWork;
  rc = fft.getBuffers(&curData, &curWork);
  std::vector<shafft::complexf> result(allocSize);
  rc = shafft::copyFromBuffer(result.data(), curWork, allocSize);

  if (rank == 0) {
    std::printf("Result[0..%d]   =", printCount - 1);
    for (int i = 0; i < printCount; ++i)
      std::printf(" (%g,%g)", result[i].real(), result[i].imag());
    std::printf("\n");
  }

  // Cleanup
  fft.release();
  rc = shafft::freeBuffer(data);
  rc = shafft::freeBuffer(work);

  MPI_Finalize();
  return 0;
}
