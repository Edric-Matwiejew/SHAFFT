/// \example example_portable_config.cpp
/// Backend-portable SHAFFT C++ example using config-driven FFTND initialization.
#include <shafft/shafft.hpp>

#include <cstdio>
#include <mpi.h>
#include <vector>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  // MPI setup
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  {
    [[maybe_unused]] int rc;
    constexpr int printCount = 4;
    std::vector<size_t> dims = {64, 64, 32};

    // Build and resolve configuration
    shafft::ConfigND cfg(dims,
                         shafft::FFTType::C2C,
                         {},
                         0,
                         shafft::DecompositionStrategy::MINIMIZE_NDA,
                         shafft::TransformLayout::REDISTRIBUTED,
                         0,
                         MPI_COMM_WORLD);

    // Create and plan FFT from config
    shafft::FFTND fft;
    rc = fft.init(cfg.cStruct());
    rc = fft.plan();

    size_t allocSize = cfg.cStruct().allocElements;
    size_t localElems = cfg.cStruct().initial.localElements;

    // Allocate buffers
    shafft::complexf *data = nullptr, *work = nullptr;
    rc = shafft::allocBuffer(allocSize, &data);
    rc = shafft::allocBuffer(allocSize, &work);

    // Initialize: delta function at origin (rank 0, index 0)
    std::vector<shafft::complexf> host(allocSize, {0.0f, 0.0f});
    if (rank == 0 && localElems > 0)
      host[0] = {1.0f, 0.0f};

    rc = shafft::copyToBuffer(data, host.data(), allocSize);
    rc = fft.setBuffers(data, work);

    // Forward FFT
    rc = fft.execute(shafft::FFTDirection::FORWARD);
    rc = fft.normalize();

    // Retrieve spectrum
    shafft::complexf *curData, *curWork;
    rc = fft.getBuffers(&curData, &curWork);
    std::vector<shafft::complexf> spectrum(allocSize);
    rc = shafft::copyFromBuffer(spectrum.data(), curData, allocSize);

    if (rank == 0) {
      std::printf("Spectrum[0..%d] =", printCount - 1);
      for (int i = 0; i < printCount; ++i)
        std::printf(" (%g,%g)", spectrum[i].real(), spectrum[i].imag());
      std::printf("\n");
    }

    // Backward FFT
    rc = fft.execute(shafft::FFTDirection::BACKWARD);
    rc = fft.normalize();

    // Retrieve result
    rc = fft.getBuffers(&curData, &curWork);
    std::vector<shafft::complexf> result(allocSize);
    rc = shafft::copyFromBuffer(result.data(), curData, allocSize);

    if (rank == 0) {
      std::printf("Result[0..%d]   =", printCount - 1);
      for (int i = 0; i < printCount; ++i)
        std::printf(" (%g,%g)", result[i].real(), result[i].imag());
      std::printf("\n");
    }

    // Cleanup
    rc = shafft::freeBuffer(data);
    rc = shafft::freeBuffer(work);
    fft.release();
  }

  MPI_Finalize();
  return 0;
}
