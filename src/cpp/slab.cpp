#include <shafft/shafft_types.hpp>
#include <shafft/shafft_config.h>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

#if SHAFFT_BACKEND_HIPFFT
#include <hip/hip_runtime.h>
#endif

// ---------------- helpers ----------------

static void decompose(int N, int M, int p, int *n, int *s) {
  int q = N / M;
  int r = N % M;
  *n = q + (r > p);
  *s = q * p + std::min(r, p);
}

static void subcomm(MPI_Comm comm, int ndims, MPI_Comm *subcomms) {
  MPI_Comm comm_cart;
  int nprocs = 0;
  MPI_Comm_size(comm, &nprocs);

  std::vector<int> dims(ndims, 0), periods(ndims, 0), remdims(ndims, 0);

  MPI_Dims_create(nprocs, ndims, dims.data());
  MPI_Cart_create(comm, ndims, dims.data(), periods.data(), 1, &comm_cart);

  int rank = 0, size = 0;
  MPI_Comm_size(comm_cart, &size);
  MPI_Comm_rank(comm_cart, &rank);

  for (int i = 0; i < ndims; i++) {
    remdims[i] = 1;
    MPI_Cart_sub(comm_cart, remdims.data(), &subcomms[i]);
    remdims[i] = 0;
  }
  MPI_Comm_free(&comm_cart);
}

static void subcomm(MPI_Comm comm, int ndims, int *cart_dims, MPI_Comm *subcomms) {
  MPI_Comm comm_cart;
  int nprocs = 0;
  MPI_Comm_size(comm, &nprocs);

  std::vector<int> periods(ndims, 0), remdims(ndims, 0);
  for (int i = 0; i < ndims; i++) {
    cart_dims[i] = 0;
  }

  MPI_Dims_create(nprocs, ndims, cart_dims);
  MPI_Cart_create(comm, ndims, cart_dims, periods.data(), 1, &comm_cart);

  int rank = 0, size = 0;
  MPI_Comm_size(comm_cart, &size);
  MPI_Comm_rank(comm_cart, &rank);

  for (int i = 0; i < ndims; i++) {
    remdims[i] = 1;
    MPI_Cart_sub(comm_cart, remdims.data(), &subcomms[i]);
    remdims[i] = 0;
  }
  MPI_Comm_free(&comm_cart);
}

static void subarray(MPI_Datatype datatype, int ndims, int *sizes, int axis,
                     int nparts, MPI_Datatype *subarrays) {
  std::vector<int> subsizes(sizes, sizes + ndims);
  std::vector<int> substarts(ndims, 0);
  int n = 0, s = 0;

  for (int p = 0; p < nparts; p++) {
    decompose(sizes[axis], nparts, p, &n, &s);
    subsizes[axis] = n;
    substarts[axis] = s;
    MPI_Type_create_subarray(ndims, sizes, subsizes.data(), substarts.data(),
                             MPI_ORDER_C, datatype, &subarrays[p]);
    MPI_Type_commit(&subarrays[p]);
  }
}

static void exchange(MPI_Comm comm, void *arrayA, MPI_Datatype *subarraysA,
                     void *arrayB, MPI_Datatype *subarraysB) {
  int nparts = 0;
  MPI_Comm_size(comm, &nparts);

  std::vector<int> counts(nparts, 1);
  std::vector<int> displs(nparts, 0);

  MPI_Alltoallw(arrayA, counts.data(), displs.data(), subarraysA,
                arrayB, counts.data(), displs.data(), subarraysB, comm);
}

static int min(int *minv, int ndim) {
  int min_ = minv[0];
  for (int i = 1; i < ndim; i++) min_ = std::min(min_, minv[i]);
  return min_;
}

static void exchange_axes(int ndim, int nda, int *da, int *ca, int *axesA,
                          int *axesB, int ith) {
  int nca = ndim - nda;
  *axesB = da[nda - ith - 1];
  *axesA = *axesB + nca;
  da[nda - ith - 1] = *axesA;
  for (int j = 0; j < nca; j++) {
    if (ca[j] == *axesA) { ca[j] = *axesB; break; }
  }
}

static int number_of_exchange_sequences(int nda, int nca) {
  return nda / nca + (nda % nca > 0);
}

static void exchange_sequences(int nda, int nca, int nes, int es[]) {
  for (int i = 0; i < nes; i++) es[i] = std::min(nca, nda - i * nca);
}

// ---------------- Slab ----------------

namespace shafft {

Slab::Slab(int ndim, int size[], int COMM_DIMS[], MPI_Datatype MPI_sendtype,
           MPI_Comm comm, size_t elem_size) {
  // Initialize runtime state
  this->_Adata = nullptr;
  this->_Bdata = nullptr;
  this->_subarrayA = nullptr;
  this->_subarrayB = nullptr;
  this->_es_index = 0;
  this->_exchange_index = 0;
  this->_max_comm_size = 0;

#if SHAFFT_BACKEND_HIPFFT && !SHAFFT_GPU_AWARE_MPI
  // Host-staging mode: initialize staging buffers
  this->_hostA = nullptr;
  this->_hostB = nullptr;
  this->_elem_size = elem_size;
#else
  (void)elem_size;  // unused when GPU-aware MPI is enabled
#endif

  this->_ndim = ndim;
  this->_worldcomm = &comm;

  // count distributed axes (leading >1)
  this->_nda = 0;
  for (int i = 0; i < ndim; i++) {
    if (COMM_DIMS[i] > 1) this->_nda++; else break;
  }
  this->_nca = ndim - this->_nda;

  // Create cartesian communicator (no reorder)
  MPI_Comm comm_cart;
  std::vector<int> periods(this->_nda, 0), remdims(this->_nda, 0);
  MPI_Cart_create(comm, this->_nda, COMM_DIMS, periods.data(), 0, &comm_cart);

  // Handle excluded ranks safely
  if (comm_cart == MPI_COMM_NULL) {
    this->_nda = 0;
    this->_nca = this->_ndim;
    this->_nes = 0;
    this->_max_comm_size = 0;

    this->_es = nullptr;
    this->_subsizes = nullptr;
    this->_offsets  = nullptr;
    this->_subsize = nullptr;
    this->_offset = nullptr;
    this->_comms = nullptr;
    this->_size = nullptr;
    this->_subarrays = nullptr;
    this->_cas = nullptr;
    this->_das = nullptr;
    this->_axesA = nullptr;
    this->_axesB = nullptr;
    this->_caA = nullptr;
    this->_daA = nullptr;

    MPI_Barrier(comm);
    return;
  }

  // create sub-communicators
  this->_comms = new MPI_Comm[this->_nda];
  int rank = 0, comm_size = 0;
  MPI_Comm_size(comm_cart, &comm_size);
  MPI_Comm_rank(comm_cart, &rank);
  for (int i = 0; i < this->_nda; i++) {
    remdims[i] = 1;
    MPI_Cart_sub(comm_cart, remdims.data(), &(this->_comms[i]));
    remdims[i] = 0;
  }
  MPI_Comm_free(&comm_cart);

  // store global tensor size
  this->_size = new int[this->_ndim];
  std::copy(size, size + this->_ndim, this->_size);

  this->_nes = number_of_exchange_sequences(this->_nda, this->_nca);
  this->_es  = new int[this->_nes];
  exchange_sequences(this->_nda, this->_nca, this->_nes, this->_es);

  this->_subsizes = new int[(this->_nda + 1) * this->_ndim];
  this->_offsets  = new int[(this->_nda + 1) * this->_ndim];

  // sizes/ranks per subcomm
  std::vector<int> comm_sizes(this->_nda), comm_ranks(this->_nda);
  this->_max_comm_size = 0;
  for (int i = 0; i < this->_nda; i++) {
    MPI_Comm_size(this->_comms[i], &comm_sizes[i]);
    MPI_Comm_rank(this->_comms[i], &comm_ranks[i]);
    this->_max_comm_size = std::max(this->_max_comm_size, comm_sizes[i]);
  }

  this->_subarrays = new MPI_Datatype[2 * this->_nda * this->_max_comm_size];

  this->_cas   = new int[this->_nca * (this->_nda + 1)];
  this->_das   = new int[this->_nda * (this->_nda + 1)];
  this->_axesA = new int[this->_nda];
  this->_axesB = new int[this->_nda];

  std::iota(this->_das, this->_das + this->_nda, 0);
  std::iota(this->_cas, this->_cas + this->_nca, this->_nda);

  for (int i = 0; i < this->_nda; i++) {
    std::copy(&(this->_das[i * this->_nda]),
              &(this->_das[i * this->_nda]) + this->_nda,
              &(this->_das[(i + 1) * this->_nda]));
    std::copy(&(this->_cas[i * this->_nca]),
              &(this->_cas[i * this->_nca]) + this->_nca,
              &(this->_cas[(i + 1) * this->_nca]));
    exchange_axes(this->_ndim, this->_nda, &(this->_das[(i + 1) * this->_nda]),
                  &(this->_cas[(i + 1) * this->_nca]), &(this->_axesA[i]),
                  &(this->_axesB[i]), i);
  }

  int n = 0, s = 0;
  for (int i = 0; i < this->_nda; i++) {
    this->_daA = &(this->_das[i * this->_nda]);
    for (int j = 0; j < this->_ndim; j++) {
      this->_subsizes[i * this->_ndim + j] = this->_size[j];
      this->_offsets [i * this->_ndim + j] = 0;
    }
    for (int j = 0; j < this->_nda; j++) {
      decompose(this->_size[this->_daA[j]], comm_sizes[j], comm_ranks[j], &n, &s);
      this->_subsizes[i * this->_ndim + this->_daA[j]] = n;
      this->_offsets [i * this->_ndim + this->_daA[j]] = s;
    }
  }

  this->_daB = &(this->_das[this->_nda * this->_nda]);
  for (int i = 0; i < this->_ndim; i++) {
    this->_subsizes[this->_nda * this->_ndim + i] = this->_size[i];
    this->_offsets [this->_nda * this->_ndim + i] = 0;
  }
  for (int j = 0; j < this->_nda; j++) {
    decompose(this->_size[this->_daB[j]], comm_sizes[j], comm_ranks[j], &n, &s);
    this->_subsizes[this->_nda * this->_ndim + this->_daB[j]] = n;
    this->_offsets [this->_nda * this->_ndim + this->_daB[j]] = s;
  }

  for (int i = 0; i < this->_nda; i++) {
    this->_subsizeA = &(this->_subsizes[i * this->_ndim]);
    this->_subsizeB = &(this->_subsizes[(i + 1) * this->_ndim]);

    subarray(MPI_sendtype, this->_ndim, this->_subsizeA, this->_axesA[i],
             comm_sizes[this->_nda - i - 1],
             &(this->_subarrays[2 * i * this->_max_comm_size]));

    subarray(MPI_sendtype, this->_ndim, this->_subsizeB, this->_axesB[i],
             comm_sizes[this->_nda - i - 1],
             &(this->_subarrays[(2 * i + 1) * this->_max_comm_size]));
  }

  this->_subsize = &(this->_subsizes[0]);
  this->_offset  = &(this->_offsets[0]);

  MPI_Barrier(comm);
}

Slab::~Slab() {
  this->_subsize = nullptr;
  this->_offset = nullptr;
  this->_subarrayA = nullptr;
  this->_subarrayB = nullptr;

  delete[] this->_size;
  delete[] this->_cas;
  delete[] this->_das;
  delete[] this->_axesA;
  delete[] this->_axesB;

  delete[] this->_es;
  delete[] this->_subsizes;
  delete[] this->_offsets;

#if SHAFFT_BACKEND_HIPFFT && !SHAFFT_GPU_AWARE_MPI
  // Free host staging buffers if allocated
  if (this->_hostA) {
    std::free(this->_hostA);
    this->_hostA = nullptr;
  }
  if (this->_hostB) {
    std::free(this->_hostB);
    this->_hostB = nullptr;
  }
#endif

  int size = 0;
  if (this->_subarrays && this->_comms) {
    for (int i = 0; i < this->_nda; i++) {
      MPI_Comm_size(this->_comms[this->_nda - i - 1], &size);
      for (int j = 0; j < size; j++) {
        MPI_Type_free(&(this->_subarrays[2 * i * this->_max_comm_size + j]));
        MPI_Type_free(&(this->_subarrays[(2 * i + 1) * this->_max_comm_size + j]));
      }
    }
  }

  if (this->_comms) {
    for (int i = 0; i < this->_nda; i++) {
      MPI_Comm_free(&this->_comms[i]);
    }
  }
  if (this->_subarrays) delete[] this->_subarrays;
  if (this->_comms)     delete[] this->_comms;
}

int Slab::get_ith_config(int *subsize, int *offset, int *ca, int ith) {
  // Handle inactive ranks: no subsizes/offsets allocated
  if (this->_subsizes == nullptr || this->_offsets == nullptr) {
    // Fill with zeros for inactive rank
    for (int i = 0; i < this->_ndim; i++) {
      subsize[i] = 0;
      offset[i] = 0;
    }
    // All axes are contiguous for inactive rank
    for (int i = 0; i < this->_nca; i++) {
      ca[i] = i;
    }
    return 0;
  }

  // Valid configs: ith in [0, _nes], where ith==_nes corresponds to "after all exchanges"
  if (ith < 0 || ith > this->_nes) {
    return static_cast<int>(shafft::Status::SHAFFT_ERR_INVALID_DIM);
  }

  int indx = 0;
  for (int i = 0; i < ith; i++) indx += this->_es[i];

  std::copy(&(this->_subsizes[indx * this->_ndim]),
            &(this->_subsizes[indx * this->_ndim]) + this->_ndim, subsize);
  std::copy(&(this->_offsets[indx * this->_ndim]),
            &(this->_offsets[indx * this->_ndim]) + this->_ndim, offset);
  std::copy(&(this->_cas[indx * this->_nca]),
            &(this->_cas[indx * this->_nca]) + this->_nca, ca);
  return 0;
}

int Slab::get_ith_layout(int *subsize, int *offset, int ith){
  // Handle inactive ranks: no subsizes/offsets allocated
  if (this->_subsizes == nullptr || this->_offsets == nullptr) {
    for (int i = 0; i < this->_ndim; i++) {
      subsize[i] = 0;
      offset[i] = 0;
    }
    return 0;
  }

  // Valid configs: ith in [0, _nes], where ith==_nes corresponds to "after all exchanges"
  if (ith < 0 || ith > this->_nes) {
    return static_cast<int>(shafft::Status::SHAFFT_ERR_INVALID_DIM);
  }

  int indx = 0;
  for (int i = 0; i < ith; i++) indx += this->_es[i];

  std::copy(&(this->_subsizes[indx * this->_ndim]),
            &(this->_subsizes[indx * this->_ndim]) + this->_ndim, subsize);
  std::copy(&(this->_offsets[indx * this->_ndim]),
            &(this->_offsets[indx * this->_ndim]) + this->_ndim, offset);
  return 0;
}

int Slab::get_ith_axes(int *ca, int *da, int ith) {
  // Valid configs: ith in [0, _nes], where ith==_nes corresponds to "after all exchanges"
  if (ith < 0 || ith > this->_nes) {
    return static_cast<int>(shafft::Status::SHAFFT_ERR_INVALID_DIM);
  }

  // Handle single-rank case: no exchanges, all axes are contiguous
  if (this->_nes == 0 || this->_cas == nullptr) {
    // Fill ca with all axes [0, 1, ..., ndim-1] (all contiguous)
    for (int i = 0; i < this->_nca; i++) {
      ca[i] = i;
    }
    // da is empty (nda == 0), nothing to copy
    return 0;
  }

  int indx = 0;
  for (int i = 0; i < ith; i++) indx += this->_es[i];

  std::copy(&(this->_cas[indx * this->_nca]),
            &(this->_cas[indx * this->_nca]) + this->_nca, ca);
  std::copy(&(this->_das[indx * this->_nda]),
            &(this->_das[indx * this->_nda]) + this->_nda, da);
  return 0;
} 

size_t Slab::alloc_size() {
  // Inactive ranks have no work
  if (this->_subsizes == nullptr) return 0;

  size_t max_elements = 0;
  for (int i = 0; i < this->_nda + 1; i++) {
    max_elements = std::max(
        max_elements,
        product<int, size_t>(&this->_subsizes[i * this->_ndim], this->_ndim));
  }
  return max_elements;
}

bool Slab::is_active() const noexcept {
  // Inactive ranks have nullptr subsizes (set in constructor when MPI_Cart_create returns MPI_COMM_NULL)
  return this->_subsizes != nullptr;
}

void Slab::set_buffers(void *Adata, void *Bdata) {
  this->_Adata = Adata;
  this->_Bdata = Bdata;
}

void Slab::get_buffers(void **Adata, void **Bdata) {
  *Adata = this->_Adata;
  *Bdata = this->_Bdata;
}

void Slab::swap_buffers() {
  void *temp = this->_Adata;
  this->_Adata = this->_Bdata;
  this->_Bdata = temp;
}

int Slab::forward() {
  if (this->_es_index < 0 || this->_es_index >= this->_nes) {
    return static_cast<int>(shafft::Status::SHAFFT_ERR_INTERNAL);
  }
  for (int i = 0; i < this->_es[this->_es_index]; i++) {
    this->prepare_forward_exchange();
    this->_exchange();
  }
  this->_es_index++;
  return 0;
}

int Slab::backward() {
  if (this->_es_index <= 0) {
    return static_cast<int>(shafft::Status::SHAFFT_ERR_INTERNAL);
  }
  for (int i = 0; i < this->_es[this->_es_index - 1]; i++) {
    this->prepare_backward_exchange();
    this->_exchange();
  }
  this->_es_index--;
  return 0;
}

int Slab::es_index() {
  return this->_es_index;
}
int Slab::nes() {
  return this->_nes;
}
void Slab::get_es(int *es) {
  std::copy(this->_es, this->_es + this->_nes, es);
}
void Slab::get_size(int *size) {
  if (this->_size == nullptr) {
    for (int i = 0; i < this->_ndim; i++) size[i] = 0;
    return;
  }
  std::copy(this->_size, this->_size + this->_ndim, size);
}
void Slab::get_subsize(int *subsize) {
  if (this->_subsize == nullptr) {
    for (int i = 0; i < this->_ndim; i++) subsize[i] = 0;
    return;
  }
  std::copy(this->_subsize, this->_subsize + this->_ndim, subsize);
}
void Slab::get_offset(int *offset) {
  if (this->_offset == nullptr) {
    for (int i = 0; i < this->_ndim; i++) offset[i] = 0;
    return;
  }
  std::copy(this->_offset, this->_offset + this->_ndim, offset);
}
void *Slab::data() { return this->_Adata; }
void *Slab::work() { return this->_Bdata; }
int Slab::ndim() { return this->_ndim; }
int Slab::nca()  { return this->_nca; }
void Slab::get_ca(int *ca) {
  // Handle single-rank case: _cas not initialized, all axes are contiguous
  if (this->_cas == nullptr || this->_nda == 0) {
    for (int i = 0; i < this->_nca; i++) {
      ca[i] = i;
    }
    return;
  }
  // Use _exchange_index to get current state from _cas array
  // _exchange_index goes from 0 to _nda as exchanges occur
  std::copy(&(this->_cas[this->_exchange_index * this->_nca]),
            &(this->_cas[this->_exchange_index * this->_nca]) + this->_nca, ca);
}
int Slab::nda()  { return this->_nda; }
void Slab::get_da(int *da) {
  // Handle single-rank case: _das not initialized, no distributed axes
  if (this->_das == nullptr || this->_nda == 0) {
    return;  // Nothing to copy
  }
  // Use _exchange_index to get current state from _das array
  std::copy(&(this->_das[this->_exchange_index * this->_nda]),
            &(this->_das[this->_exchange_index * this->_nda]) + this->_nda, da);
}

void Slab::prepare_forward_exchange() {
  assert((this->_exchange_index >= 0 && this->_exchange_index < this->_nda) &&
         "exchange index out of range");

  this->_comm = this->_comms[this->_nda - this->_exchange_index - 1];
  this->_subsizeA = &(this->_subsizes[this->_exchange_index * this->_ndim]);
  this->_subsizeB = &(this->_subsizes[(this->_exchange_index + 1) * this->_ndim]);
  this->_subarrayA = &(this->_subarrays[2 * this->_exchange_index * this->_max_comm_size]);
  this->_subarrayB = &(this->_subarrays[(2 * this->_exchange_index + 1) * this->_max_comm_size]);

  this->_taA = this->_axesA[this->_exchange_index];
  this->_taB = this->_axesB[this->_exchange_index];

  this->_daA = &(this->_das[this->_exchange_index * this->_nda]);
  this->_caA = &(this->_cas[this->_exchange_index * this->_nca]);
  this->_daB = &(this->_das[(this->_exchange_index + 1) * this->_nda]);
  this->_caB = &(this->_cas[(this->_exchange_index + 1) * this->_nca]);

  this->_exchange_direction = 1;
}

void Slab::prepare_backward_exchange() {
  // For backward: _exchange_index starts at _nda and decrements to 1
  // Array accesses use (_exchange_index - 1) which maps to [0, _nda)
  assert((this->_exchange_index > 0 && this->_exchange_index <= this->_nda) &&
         "exchange index out of range");

  this->_comm = this->_comms[this->_nda - this->_exchange_index];

  this->_subsizeA = &(this->_subsizes[(this->_exchange_index) * this->_ndim]);
  this->_subsizeB = &(this->_subsizes[(this->_exchange_index - 1) * this->_ndim]);
  this->_subarrayA = &(this->_subarrays[(2 * (this->_exchange_index - 1) + 1) * this->_max_comm_size]);
  this->_subarrayB = &(this->_subarrays[2 * (this->_exchange_index - 1) * this->_max_comm_size]);

  this->_taA = this->_axesB[this->_exchange_index - 1];
  this->_taB = this->_axesA[this->_exchange_index - 1];

  this->_daA = &(this->_das[this->_exchange_index * this->_nda]);
  this->_caA = &(this->_cas[this->_exchange_index * this->_nca]);
  this->_daB = &(this->_das[(this->_exchange_index - 1) * this->_nda]);
  this->_caB = &(this->_cas[(this->_exchange_index - 1) * this->_nca]);

  this->_exchange_direction = -1;
}

void Slab::_exchange() {
  if (this->_Adata == nullptr || this->_Bdata == nullptr) {
    std::cerr << "Error: Missing buffers." << std::endl;
    return;
  }

#if SHAFFT_BACKEND_HIPFFT && !SHAFFT_GPU_AWARE_MPI
  // Host-staging mode: copy device -> host, MPI exchange, copy host -> device
  {
    size_t alloc_elems = this->alloc_size();
    size_t buf_bytes = alloc_elems * this->_elem_size;

    // Lazily allocate host staging buffers
    if (this->_hostA == nullptr) {
      this->_hostA = std::malloc(buf_bytes);
      if (!this->_hostA) {
        std::cerr << "Error: Failed to allocate host staging buffer A (" 
                  << buf_bytes << " bytes)" << std::endl;
        return;
      }
    }
    if (this->_hostB == nullptr) {
      this->_hostB = std::malloc(buf_bytes);
      if (!this->_hostB) {
        std::cerr << "Error: Failed to allocate host staging buffer B (" 
                  << buf_bytes << " bytes)" << std::endl;
        return;
      }
    }

    // Copy input from device to host
    hipError_t err = hipMemcpy(this->_hostA, this->_Adata, buf_bytes, hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
      std::cerr << "Error: hipMemcpy D2H failed: " << hipGetErrorString(err) << std::endl;
      return;
    }

    // Do MPI exchange on host buffers
    exchange(this->_comm, this->_hostA, this->_subarrayA, this->_hostB, this->_subarrayB);

    // Copy result back to device
    err = hipMemcpy(this->_Bdata, this->_hostB, buf_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
      std::cerr << "Error: hipMemcpy H2D failed: " << hipGetErrorString(err) << std::endl;
      return;
    }
  }
#else
  // GPU-aware MPI (or FFTW backend): exchange directly on buffer pointers
  exchange(this->_comm, this->_Adata, this->_subarrayA, this->_Bdata, this->_subarrayB);
#endif

  this->swap_buffers();

  this->_exchange_index += this->_exchange_direction;

  this->_subsize = &(this->_subsizes[this->_exchange_index * this->_ndim]);
  this->_offset  = &(this->_offsets [this->_exchange_index * this->_ndim]);
}

void Slab::get_indices(int index, int ndim, int *size, int *offset, int *indices) {
  for (int i = ndim - 1; i >= 0; i--) {
    indices[i] = index % size[i] + offset[i];
    index /= size[i];
  }
}

int Slab::get_index(int *indices, int ndim, int *size) {
  std::vector<int> strides(ndim, 0);
  strides[ndim - 1] = 1;
  for (int i = ndim - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * size[i + 1];
  }
  int index = 0;
  for (int i = 0; i < ndim; i++) {
    index += (indices[i]) * strides[i];
  }
  return index;
}

} // namespace shafft
