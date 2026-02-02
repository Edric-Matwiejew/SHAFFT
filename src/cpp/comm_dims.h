#ifndef COMM_DIMS_H
#define COMM_DIMS_H

#include <vector>

// Create a grid size for a Cartesian communicator for a tensor with nda
// distributed axes
int create_comm_dims(int ndim, int COMM_SIZE, int nda, const int* sizes, int* COMM_DIMS);

#endif  // COMM_DIMS_H