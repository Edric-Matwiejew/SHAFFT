/** \example example.cpp
 *  SHAFFT C++ example using HIP directly.
 *  Demonstrates the procedural API with explicit GPU memory management.
 */
#include <shafft/shafft.hpp>
#include <mpi.h>
#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <vector>
#include <iostream>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int ndim = 3;
  std::vector<int> dims = {64, 64, 32};

  // Create plan (procedural API uses PlanData*)
  shafft::PlanData* plan; 
  shafft::planCreate(&plan);

  // NDA planner
  shafft::planNDA(plan, 1, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);

  // --- Alternative: Cartesian planner ---
  /*
  std::vector<int> subsize(ndim), offset(ndim), COMM_DIMS(ndim, 0);
  int COMM_SIZE;
  size_t mem_limit = 0;
  shafft::configurationCart(dims, subsize, offset, COMM_DIMS, COMM_SIZE,
                            shafft::FFTType::C2C, mem_limit, MPI_COMM_WORLD);
  shafft::planCart(plan, COMM_DIMS, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
  */

  // Get decomposition info
  std::vector<int> subsize(ndim), offset(ndim);
  shafft::getLayout(plan, subsize, offset, shafft::TensorLayout::CURRENT);

  size_t alloc_size = 0; 
  shafft::getAllocSize(plan, alloc_size);

  // Allocate device memory
  hipComplex* d_data;
  hipComplex* d_work;
  hipMalloc(&d_data, alloc_size * sizeof(hipComplex));
  hipMalloc(&d_work, alloc_size * sizeof(hipComplex));

  // Initialise host memory: set h[0] = (1,0) on rank 0 only, zeros elsewhere
  int local_elems = 1;
  for (int i = 0; i < ndim; ++i) local_elems *= subsize[i];
  std::vector<hipComplex> h(local_elems, {0.0f, 0.0f});
  if (rank == 0) {
    h[0].x = 1.0f;
    h[0].y = 0.0f;
  }

  // Print initial data
  if (rank == 0) {
    std::cout << "=== Initial data ===" << std::endl;
    std::cout << "Global dims: " << dims[0] << "x" << dims[1] << "x" << dims[2] << std::endl;
  }
  for (int r = 0; r < size; ++r) {
    if (rank == r) {
      std::cout << "Rank " << rank << " subsize: " 
                << subsize[0] << "x" << subsize[1] << "x" << subsize[2]
                << ", offset: " << offset[0] << "," << offset[1] << "," << offset[2]
                << ", h[0] = (" << h[0].x << ", " << h[0].y << ")" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  hipMemcpy(d_data, h.data(), local_elems * sizeof(hipComplex),
            hipMemcpyHostToDevice);

  // Attach buffers and execute FFT
  shafft::setBuffers(plan, d_data, d_work);
  shafft::execute(plan, shafft::FFTDirection::FORWARD);
  shafft::normalize(plan);
  shafft::getBuffers(plan, &d_data, &d_work);

  // Copy back to check forward result
  hipMemcpy(h.data(), d_data, local_elems * sizeof(hipComplex),
            hipMemcpyDeviceToHost);

  // Print result: FFT of delta function scaled by 1/sqrt(N) gives constant 1/sqrt(N)
  if (rank == 0) {
    std::cout << "\n=== After forward FFT + normalize ===" << std::endl;
    int N = dims[0] * dims[1] * dims[2];
    float expected = 1.0f / std::sqrt(static_cast<float>(N));
    std::cout << "Expected: constant (" << expected << ", 0) everywhere (symmetric normalization)" << std::endl;
  }
  for (int r = 0; r < size; ++r) {
    if (rank == r) {
      std::cout << "Rank " << rank << " first 4 values: ";
      for (int i = 0; i < std::min(4, local_elems); ++i) {
        std::cout << "(" << h[i].x << "," << h[i].y << ") ";
      }
      std::cout << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Now do backward FFT to recover original data
  shafft::setBuffers(plan, d_data, d_work);
  shafft::execute(plan, shafft::FFTDirection::BACKWARD);
  shafft::normalize(plan);
  shafft::getBuffers(plan, &d_data, &d_work);

  // Copy back
  hipMemcpy(h.data(), d_data, local_elems * sizeof(hipComplex),
            hipMemcpyDeviceToHost);

  // Print result: should recover original delta function
  if (rank == 0) {
    std::cout << "\n=== After backward FFT + normalize ===" << std::endl;
    std::cout << "Expected: delta at origin - rank 0 h[0]=(1,0), all others=(0,0)" << std::endl;
  }
  for (int r = 0; r < size; ++r) {
    if (rank == r) {
      std::cout << "Rank " << rank << " first 4 values: ";
      for (int i = 0; i < std::min(4, local_elems); ++i) {
        std::cout << "(" << h[i].x << "," << h[i].y << ") ";
      }
      std::cout << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  hipFree(d_data);
  hipFree(d_work);
  shafft::destroy(&plan);

  if (rank == 0) {
    std::cout << "\nSHAFFT HIP example completed successfully with " << size << " MPI ranks" << std::endl;
  }

  MPI_Finalize();
  return 0;
}
