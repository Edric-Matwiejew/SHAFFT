/** \example example_portable.cpp
 *  Backend-agnostic SHAFFT C++ example.
 *  Uses shafft::complexf and portable buffer functions so the same code
 *  works on both CPU (FFTW) and GPU (hipFFT) backends.
 */
#include <shafft/shafft.hpp>
#include <mpi.h>
#include <vector>
#include <cstdio>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Print library info on rank 0
  if (rank == 0) {
    std::printf("SHAFFT %s (backend: %s)\n",
                shafft::getVersionString(), shafft::getBackendName());
  }

  const int ndim = 3;
  std::vector<int> dims = {64, 64, 32};

  // Create and initialize plan
  shafft::Plan plan;
  plan.init(1, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);

  // Get local layout
  std::vector<int> subsize(ndim), offset(ndim);
  plan.getLayout(subsize, offset, shafft::TensorLayout::CURRENT);

  size_t alloc_elems = plan.allocSize();
  int local_elems = subsize[0] * subsize[1] * subsize[2];

  // Allocate portable buffers (device memory on GPU, host on CPU)
  shafft::complexf* data = nullptr;
  shafft::complexf* work = nullptr;
  shafft::allocBuffer(alloc_elems, &data);
  shafft::allocBuffer(alloc_elems, &work);

  // Initialize host data with a simple impulse
  std::vector<shafft::complexf> host_data(alloc_elems, {0.0f, 0.0f});
  if (local_elems > 0) {
    host_data[0] = {1.0f, 1.0f};
  }

  // Copy to compute buffer and attach to plan
  shafft::copyToBuffer(data, host_data.data(), alloc_elems);
  plan.setBuffers(data, work);

  // Execute forward FFT and normalize
  plan.execute(shafft::FFTDirection::FORWARD);
  plan.normalize();

  // Get current buffer (may have been swapped internally)
  shafft::complexf* cur_data = nullptr;
  shafft::complexf* cur_work = nullptr;
  plan.getBuffers(&cur_data, &cur_work);

  // Copy result back to host
  std::vector<shafft::complexf> result(alloc_elems);
  shafft::copyFromBuffer(result.data(), cur_data, alloc_elems);

  // Print a few values on rank 0
  if (rank == 0) {
    std::printf("Portable example completed.\n");
    if (local_elems >= 4) {
      std::printf("Result[0..3] = (%g,%g) (%g,%g) (%g,%g) (%g,%g)\n",
        result[0].real(), result[0].imag(),
        result[1].real(), result[1].imag(),
        result[2].real(), result[2].imag(),
        result[3].real(), result[3].imag());
    }
  }

  // Cleanup
  shafft::freeBuffer(data);
  shafft::freeBuffer(work);
  plan.release();

  MPI_Finalize();
  return 0;
}
