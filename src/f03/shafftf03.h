#ifndef SHAFFT_F03_H
#define SHAFFT_F03_H

#include <shafft/shafft.h>

#include <mpi.h>

extern "C" {

int shafftConfigurationNDAf03(int ndim, int* size, int* nda, int* subsize, int* offset,
                              int* COMM_DIMS, shafft_t precision, size_t mem_limit,
                              MPI_Fint* f_handle);

int shafftConfigurationCartf03(int ndim, int* size, int* subsize, int* offset, int* COMM_DIMS,
                               int* COMM_SIZE, shafft_t precision, size_t mem_limit,
                               MPI_Fint* f_handle);

int shafftPlanNDAf03(void* plan_ptr, int ndim, int nda, int dimensions[], shafft_t precision,
                     MPI_Fint* f_handle);

int shafftPlanCartf03(void* plan_ptr, int ndim, int COMM_DIMS[], int dimensions[],
                      shafft_t precision, MPI_Fint* f_handle);
}

#endif  // SHAFFT_F03_H
