#ifndef SHAFFT_H
#define SHAFFT_H

#include <shafft/shafft_config.h>
#include <shafft/shafft_error.hpp>
#include <shafft/shafft_types.hpp>

#include <cstdlib>
#include <mpi.h>

#if SHAFFT_BACKEND_HIPFFT
#include <hip/hip_runtime.h>
#endif

struct fftHandle;

namespace _shafft {

#if SHAFFT_BACKEND_HIPFFT
int setStream(shafft::PlanData* plan, hipStream_t stream) noexcept;
#endif

int planCreate(shafft::PlanData** plan) noexcept;

template <typename T>
int setBuffers(shafft::PlanData* plan, T* data, T* work) noexcept {
  try {
    if (!plan)
      return (int)shafft::Status::SHAFFT_ERR_NULLPTR;
    if (!plan->slab)
      return (int)shafft::Status::SHAFFT_ERR_PLAN_NOT_INIT;
    if (!data || !work)
      return (int)shafft::Status::SHAFFT_ERR_NULLPTR;

    plan->slab->set_buffers(static_cast<void*>(data), static_cast<void*>(work));
    return static_cast<int>(shafft::Status::SHAFFT_SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

template <typename T>
int getBuffers(shafft::PlanData* plan, T** data, T** work) noexcept {
  try {
    if (!plan)
      return (int)shafft::Status::SHAFFT_ERR_NULLPTR;
    if (!plan->slab)
      return (int)shafft::Status::SHAFFT_ERR_PLAN_NOT_INIT;
    if (!data || !work)
      return (int)shafft::Status::SHAFFT_ERR_NULLPTR;

    void* d = nullptr;
    void* w = nullptr;
    plan->slab->get_buffers(&d, &w);
    if (!d || !w)
      return (int)shafft::Status::SHAFFT_ERR_NO_BUFFER;

    *data = static_cast<T*>(d);
    *work = static_cast<T*>(w);
    return static_cast<int>(shafft::Status::SHAFFT_SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

int configurationNDA(int ndim, int* size, int* nda, int* subsize, int* offset, int* COMM_DIMS,
                     shafft::FFTType precision, size_t mem_limit, MPI_Comm COMM) noexcept;

int configurationCart(int ndim, int* size, int* subsize, int* offset, int* COMM_DIMS,
                      int* COMM_SIZE, shafft::FFTType precision, size_t mem_limit,
                      MPI_Comm COMM) noexcept;

int planNDA(shafft::PlanData* plan, int ndim, int nda, int dimensions[], shafft::FFTType precision,
            MPI_Comm COMM) noexcept;

int planCart(shafft::PlanData* plan, int ndim, int COMM_DIMS[], int dimensions[],
             shafft::FFTType precision, MPI_Comm COMM) noexcept;

int destroy(shafft::PlanData** plan) noexcept;

int getLayout(shafft::PlanData* plan, int* subsize, int* offset,
              shafft::TensorLayout layout) noexcept;

int getAxes(shafft::PlanData* plan, int* ca, int* da, shafft::TensorLayout layout) noexcept;

int getAllocSize(shafft::PlanData* plan, size_t* alloc_size) noexcept;

int execute(shafft::PlanData* plan, shafft::FFTDirection direction) noexcept;

int normalize(shafft::PlanData* plan) noexcept;

}  // namespace _shafft

#endif  // SHAFFT_H
