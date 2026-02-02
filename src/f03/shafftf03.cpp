#include "shafftf03.h"

#include <shafft/shafft.h>

int shafftConfigurationNDAf03(int ndim, int* size, int* nda, int* subsize, int* offset,
                              int* COMM_DIMS, shafft_t precision, size_t mem_limit,
                              MPI_Fint* f_handle) {
  MPI_Comm c_comm = MPI_Comm_f2c(*f_handle);
  return shafftConfigurationNDA(ndim, size, nda, subsize, offset, COMM_DIMS, precision, mem_limit,
                                c_comm);
}

int shafftConfigurationCartf03(int ndim, int* size, int* subsize, int* offset, int* COMM_DIMS,
                               int* COMM_SIZE, shafft_t precision, size_t mem_limit,
                               MPI_Fint* f_handle) {
  MPI_Comm c_comm = MPI_Comm_f2c(*f_handle);
  return shafftConfigurationCart(ndim, size, subsize, offset, COMM_DIMS, COMM_SIZE, precision,
                                 mem_limit, c_comm);
}

int shafftPlanNDAf03(void* plan_ptr, int ndim, int nda, int dimensions[], shafft_t precision,
                     MPI_Fint* f_handle) {
  MPI_Comm c_comm = MPI_Comm_f2c(*f_handle);
  return shafftPlanNDA(plan_ptr, ndim, nda, dimensions, precision, c_comm);
}

int shafftPlanCartf03(void* plan_ptr, int ndim, int COMM_DIMS[], int dimensions[],
                      shafft_t precision, MPI_Fint* f_handle) {
  MPI_Comm c_comm = MPI_Comm_f2c(*f_handle);
  return shafftPlanCart(plan_ptr, ndim, COMM_DIMS, dimensions, precision, c_comm);
}
