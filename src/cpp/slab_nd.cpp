#include "detail/slab_nd.hpp"
#include "detail/cartesian.hpp"
#include "detail/partition.hpp"
#include <shafft/shafft_config.h>
#include <shafft/shafft_types.hpp>

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

using shafft::detail::block_partition;
using shafft::detail::make_subcomms;

static void subarray(
    MPI_Datatype datatype, int ndims, int* sizes, int axis, int nparts, MPI_Datatype* subarrays) {
  std::vector<int> subsizes(sizes, sizes + ndims);
  std::vector<int> substarts(ndims, 0);
  int n = 0, s = 0;

  for (int p = 0; p < nparts; p++) {
    block_partition(sizes[axis], nparts, p, &n, &s);
    subsizes[axis] = n;
    substarts[axis] = s;
    MPI_Type_create_subarray(
        ndims, sizes, subsizes.data(), substarts.data(), MPI_ORDER_C, datatype, &subarrays[p]);
    MPI_Type_commit(&subarrays[p]);
  }
}

static void exchange(
    MPI_Comm comm, void* arrayA, MPI_Datatype* subarraysA, void* arrayB, MPI_Datatype* subarraysB) {
  int nparts = 0;
  MPI_Comm_size(comm, &nparts);

  std::vector<int> counts(nparts, 1);
  std::vector<int> displs(nparts, 0);

  MPI_Alltoallw(arrayA,
                counts.data(),
                displs.data(),
                subarraysA,
                arrayB,
                counts.data(),
                displs.data(),
                subarraysB,
                comm);
}

static void exchangeAxes(int ndim, int nda, int* da, int* ca, int* axesA, int* axesB, int ith) {
  int nca = ndim - nda;
  *axesB = da[nda - ith - 1];
  *axesA = *axesB + nca;
  da[nda - ith - 1] = *axesA;
  for (int j = 0; j < nca; j++) {
    if (ca[j] == *axesA) {
      ca[j] = *axesB;
      break;
    }
  }
}

static int numberOfExchangeSequences(int nda, int nca) {
  return nda / nca + (nda % nca > 0);
}

static void exchangeSequences(int nda, int nca, int nes, int es[]) {
  for (int i = 0; i < nes; i++)
    es[i] = std::min(nca, nda - i * nca);
}

// ---------------- SlabND ----------------

namespace shafft {

SlabND::SlabND(int ndim,
               int size[],
               int commDims[],
               MPI_Datatype mpiSendtype,
               MPI_Comm comm,
               size_t elemSize) {
  // Initialize runtime state
  this->aData_ = nullptr;
  this->bData_ = nullptr;
  this->subarrayA_ = nullptr;
  this->subarrayB_ = nullptr;
  this->esIndex_ = 0;
  this->exchangeIndex_ = 0;
  this->maxCommSize_ = 0;

#if SHAFFT_BACKEND_HIPFFT && !SHAFFT_GPU_AWARE_MPI
  // Host-staging mode: initialize staging buffers
  this->hostA_ = nullptr;
  this->hostB_ = nullptr;
  this->elemSize_ = elemSize;
#else
  (void)elemSize; // unused when GPU-aware MPI is enabled
#endif

  this->ndim_ = ndim;
  this->worldcomm_ = &comm;

  // count distributed axes (leading >1)
  this->nda_ = 0;
  for (int i = 0; i < ndim; i++) {
    if (commDims[i] > 1)
      this->nda_++;
    else
      break;
  }
  this->nca_ = ndim - this->nda_;

  // Create cartesian communicator (no reorder)
  MPI_Comm commCart;
  std::vector<int> periods(this->nda_, 0);
  MPI_Cart_create(comm, this->nda_, commDims, periods.data(), 0, &commCart);

  // Handle excluded ranks safely
  if (commCart == MPI_COMM_NULL) {
    this->nda_ = 0;
    this->nca_ = this->ndim_;
    this->nes_ = 0;
    this->maxCommSize_ = 0;

    this->es_ = nullptr;
    this->subsizes_ = nullptr;
    this->offsets_ = nullptr;
    this->subsize_ = nullptr;
    this->offset_ = nullptr;
    this->comms_ = nullptr;
    this->size_ = nullptr;
    this->subarrays_ = nullptr;
    this->cas_ = nullptr;
    this->das_ = nullptr;
    this->axesA_ = nullptr;
    this->axesB_ = nullptr;
    this->caA_ = nullptr;
    this->daA_ = nullptr;

    MPI_Barrier(comm);
    return;
  }

  // create sub-communicators
  this->comms_ = new MPI_Comm[this->nda_];
  int rank = 0, commSize = 0;
  MPI_Comm_size(commCart, &commSize);
  MPI_Comm_rank(commCart, &rank);
  make_subcomms(commCart, this->nda_, this->comms_);
  MPI_Comm_free(&commCart);

  // store global tensor size
  this->size_ = new int[this->ndim_];
  std::copy(size, size + this->ndim_, this->size_);

  this->nes_ = numberOfExchangeSequences(this->nda_, this->nca_);
  this->es_ = new int[this->nes_];
  exchangeSequences(this->nda_, this->nca_, this->nes_, this->es_);

  this->subsizes_ = new int[(this->nda_ + 1) * this->ndim_];
  this->offsets_ = new int[(this->nda_ + 1) * this->ndim_];

  // sizes/ranks per subcomm
  std::vector<int> commSizes(this->nda_), commRanks(this->nda_);
  this->maxCommSize_ = 0;
  for (int i = 0; i < this->nda_; i++) {
    MPI_Comm_size(this->comms_[i], &commSizes[i]);
    MPI_Comm_rank(this->comms_[i], &commRanks[i]);
    this->maxCommSize_ = std::max(this->maxCommSize_, commSizes[i]);
  }

  this->subarrays_ = new MPI_Datatype[2 * this->nda_ * this->maxCommSize_];

  this->cas_ = new int[this->nca_ * (this->nda_ + 1)];
  this->das_ = new int[this->nda_ * (this->nda_ + 1)];
  this->axesA_ = new int[this->nda_];
  this->axesB_ = new int[this->nda_];

  std::iota(this->das_, this->das_ + this->nda_, 0);
  std::iota(this->cas_, this->cas_ + this->nca_, this->nda_);

  for (int i = 0; i < this->nda_; i++) {
    std::copy(&(this->das_[i * this->nda_]),
              &(this->das_[i * this->nda_]) + this->nda_,
              &(this->das_[(i + 1) * this->nda_]));
    std::copy(&(this->cas_[i * this->nca_]),
              &(this->cas_[i * this->nca_]) + this->nca_,
              &(this->cas_[(i + 1) * this->nca_]));
    exchangeAxes(this->ndim_,
                 this->nda_,
                 &(this->das_[(i + 1) * this->nda_]),
                 &(this->cas_[(i + 1) * this->nca_]),
                 &(this->axesA_[i]),
                 &(this->axesB_[i]),
                 i);
  }

  int n = 0, s = 0;
  for (int i = 0; i < this->nda_; i++) {
    this->daA_ = &(this->das_[i * this->nda_]);
    for (int j = 0; j < this->ndim_; j++) {
      this->subsizes_[i * this->ndim_ + j] = this->size_[j];
      this->offsets_[i * this->ndim_ + j] = 0;
    }
    for (int j = 0; j < this->nda_; j++) {
      block_partition(this->size_[this->daA_[j]], commSizes[j], commRanks[j], &n, &s);
      this->subsizes_[i * this->ndim_ + this->daA_[j]] = n;
      this->offsets_[i * this->ndim_ + this->daA_[j]] = s;
    }
  }

  this->daB_ = &(this->das_[this->nda_ * this->nda_]);
  for (int i = 0; i < this->ndim_; i++) {
    this->subsizes_[this->nda_ * this->ndim_ + i] = this->size_[i];
    this->offsets_[this->nda_ * this->ndim_ + i] = 0;
  }
  for (int j = 0; j < this->nda_; j++) {
    block_partition(this->size_[this->daB_[j]], commSizes[j], commRanks[j], &n, &s);
    this->subsizes_[this->nda_ * this->ndim_ + this->daB_[j]] = n;
    this->offsets_[this->nda_ * this->ndim_ + this->daB_[j]] = s;
  }

  for (int i = 0; i < this->nda_; i++) {
    this->subsizeA_ = &(this->subsizes_[i * this->ndim_]);
    this->subsizeB_ = &(this->subsizes_[(i + 1) * this->ndim_]);

    subarray(mpiSendtype,
             this->ndim_,
             this->subsizeA_,
             this->axesA_[i],
             commSizes[this->nda_ - i - 1],
             &(this->subarrays_[2 * i * this->maxCommSize_]));

    subarray(mpiSendtype,
             this->ndim_,
             this->subsizeB_,
             this->axesB_[i],
             commSizes[this->nda_ - i - 1],
             &(this->subarrays_[(2 * i + 1) * this->maxCommSize_]));
  }

  this->subsize_ = &(this->subsizes_[0]);
  this->offset_ = &(this->offsets_[0]);

  MPI_Barrier(comm);
}

SlabND::~SlabND() {
  this->subsize_ = nullptr;
  this->offset_ = nullptr;
  this->subarrayA_ = nullptr;
  this->subarrayB_ = nullptr;

  delete[] this->size_;
  delete[] this->cas_;
  delete[] this->das_;
  delete[] this->axesA_;
  delete[] this->axesB_;

  delete[] this->es_;
  delete[] this->subsizes_;
  delete[] this->offsets_;

#if SHAFFT_BACKEND_HIPFFT && !SHAFFT_GPU_AWARE_MPI
  // Free host staging buffers if allocated
  if (this->hostA_) {
    std::free(this->hostA_);
    this->hostA_ = nullptr;
  }
  if (this->hostB_) {
    std::free(this->hostB_);
    this->hostB_ = nullptr;
  }
#endif

  int size = 0;
  if (this->subarrays_ && this->comms_) {
    for (int i = 0; i < this->nda_; i++) {
      MPI_Comm_size(this->comms_[this->nda_ - i - 1], &size);
      for (int j = 0; j < size; j++) {
        MPI_Type_free(&(this->subarrays_[2 * i * this->maxCommSize_ + j]));
        MPI_Type_free(&(this->subarrays_[(2 * i + 1) * this->maxCommSize_ + j]));
      }
    }
  }

  if (this->comms_) {
    for (int i = 0; i < this->nda_; i++) {
      MPI_Comm_free(&this->comms_[i]);
    }
  }
  if (this->subarrays_)
    delete[] this->subarrays_;
  if (this->comms_)
    delete[] this->comms_;
}

int SlabND::getIthConfig(int* subsize, int* offset, int* ca, int ith) const {
  // Handle inactive ranks: no subsizes/offsets allocated
  if (this->subsizes_ == nullptr || this->offsets_ == nullptr) {
    // Fill with zeros for inactive rank
    for (int i = 0; i < this->ndim_; i++) {
      subsize[i] = 0;
      offset[i] = 0;
    }
    // All axes are contiguous for inactive rank
    for (int i = 0; i < this->nca_; i++) {
      ca[i] = i;
    }
    return 0;
  }

  // Valid configs: ith in [0, nes_], where ith==nes_ corresponds to "after all exchanges"
  if (ith < 0 || ith > this->nes_) {
    return static_cast<int>(shafft::Status::ERR_INVALID_DIM);
  }

  int indx = 0;
  for (int i = 0; i < ith; i++)
    indx += this->es_[i];

  std::copy(&(this->subsizes_[indx * this->ndim_]),
            &(this->subsizes_[indx * this->ndim_]) + this->ndim_,
            subsize);
  std::copy(&(this->offsets_[indx * this->ndim_]),
            &(this->offsets_[indx * this->ndim_]) + this->ndim_,
            offset);
  std::copy(&(this->cas_[indx * this->nca_]), &(this->cas_[indx * this->nca_]) + this->nca_, ca);
  return 0;
}

int SlabND::getIthLayout(int* subsize, int* offset, int ith) const {
  // Handle inactive ranks: no subsizes/offsets allocated
  if (this->subsizes_ == nullptr || this->offsets_ == nullptr) {
    for (int i = 0; i < this->ndim_; i++) {
      subsize[i] = 0;
      offset[i] = 0;
    }
    return 0;
  }

  // Valid configs: ith in [0, nes_], where ith==nes_ corresponds to "after all exchanges"
  if (ith < 0 || ith > this->nes_) {
    return static_cast<int>(shafft::Status::ERR_INVALID_DIM);
  }

  int indx = 0;
  for (int i = 0; i < ith; i++)
    indx += this->es_[i];

  std::copy(&(this->subsizes_[indx * this->ndim_]),
            &(this->subsizes_[indx * this->ndim_]) + this->ndim_,
            subsize);
  std::copy(&(this->offsets_[indx * this->ndim_]),
            &(this->offsets_[indx * this->ndim_]) + this->ndim_,
            offset);
  return 0;
}

int SlabND::getIthAxes(int* ca, int* da, int ith) const noexcept {
  // Valid configs: ith in [0, nes_], where ith==nes_ corresponds to "after all exchanges"
  if (ith < 0 || ith > this->nes_) {
    return static_cast<int>(shafft::Status::ERR_INVALID_DIM);
  }

  // Handle single-rank case: no exchanges, all axes are contiguous
  if (this->nes_ == 0 || this->cas_ == nullptr) {
    // Fill ca with all axes [0, 1, ..., ndim-1] (all contiguous)
    for (int i = 0; i < this->nca_; i++) {
      ca[i] = i;
    }
    // da is empty (nda == 0), nothing to copy
    return 0;
  }

  int indx = 0;
  for (int i = 0; i < ith; i++)
    indx += this->es_[i];

  std::copy(&(this->cas_[indx * this->nca_]), &(this->cas_[indx * this->nca_]) + this->nca_, ca);
  std::copy(&(this->das_[indx * this->nda_]), &(this->das_[indx * this->nda_]) + this->nda_, da);
  return 0;
}

size_t SlabND::localAllocSize() const {
  // Inactive ranks have no work
  if (this->subsizes_ == nullptr)
    return 0;

  size_t maxElements = 0;
  for (int i = 0; i < this->nda_ + 1; i++) {
    maxElements =
        std::max(maxElements, product<int, size_t>(&this->subsizes_[i * this->ndim_], this->ndim_));
  }
  return maxElements;
}

bool SlabND::isActive() const noexcept {
  // Inactive ranks have nullptr subsizes (set in constructor when MPI_Cart_create returns
  // MPI_COMM_NULL)
  return this->subsizes_ != nullptr;
}

// SlabBase interface implementations

size_t SlabND::allocSize() const noexcept {
  return localAllocSize();
}

int SlabND::getIthLayout(size_t* subsize, size_t* offset, int i) const noexcept {
  // Temporary int arrays for existing getIthLayout (value-initialized to zero)
  int* subsizeInt = new int[ndim_]();
  int* offsetInt = new int[ndim_]();

  int result = getIthLayout(subsizeInt, offsetInt, i);

  // Only convert to size_t if successful
  if (result == 0) {
    for (int j = 0; j < ndim_; ++j) {
      subsize[j] = static_cast<size_t>(subsizeInt[j]);
      offset[j] = static_cast<size_t>(offsetInt[j]);
    }
  }

  delete[] subsizeInt;
  delete[] offsetInt;
  return result;
}

void SlabND::setBuffers(void* aData, void* bData) noexcept {
  this->aData_ = aData;
  this->bData_ = bData;
}

void SlabND::getBuffers(void** aData, void** bData) const noexcept {
  *aData = this->aData_;
  *bData = this->bData_;
}

void SlabND::swapBuffers() {
  void* temp = this->aData_;
  this->aData_ = this->bData_;
  this->bData_ = temp;
}

int SlabND::forward() {
  if (this->esIndex_ < 0 || this->esIndex_ >= this->nes_) {
    return static_cast<int>(shafft::Status::ERR_INTERNAL);
  }
  for (int i = 0; i < this->es_[this->esIndex_]; i++) {
    this->prepareForwardExchange();
    this->doExchange();
  }
  this->esIndex_++;
  return 0;
}

int SlabND::backward() {
  if (this->esIndex_ <= 0) {
    return static_cast<int>(shafft::Status::ERR_INTERNAL);
  }
  for (int i = 0; i < this->es_[this->esIndex_ - 1]; i++) {
    this->prepareBackwardExchange();
    this->doExchange();
  }
  this->esIndex_--;
  return 0;
}

int SlabND::esIndex() {
  return this->esIndex_;
}
int SlabND::nes() {
  return this->nes_;
}
void SlabND::getEs(int* es) {
  std::copy(this->es_, this->es_ + this->nes_, es);
}
void SlabND::getSize(int* size) {
  if (this->size_ == nullptr) {
    for (int i = 0; i < this->ndim_; i++)
      size[i] = 0;
    return;
  }
  std::copy(this->size_, this->size_ + this->ndim_, size);
}
void SlabND::getSubsize(int* subsize) {
  if (this->subsize_ == nullptr) {
    for (int i = 0; i < this->ndim_; i++)
      subsize[i] = 0;
    return;
  }
  std::copy(this->subsize_, this->subsize_ + this->ndim_, subsize);
}
void SlabND::getOffset(int* offset) {
  if (this->offset_ == nullptr) {
    for (int i = 0; i < this->ndim_; i++)
      offset[i] = 0;
    return;
  }
  std::copy(this->offset_, this->offset_ + this->ndim_, offset);
}
void* SlabND::data() {
  return this->aData_;
}
void* SlabND::work() {
  return this->bData_;
}

int SlabND::nca() const noexcept {
  return this->nca_;
}
void SlabND::getCa(int* ca) const noexcept {
  // Handle single-rank case: cas_ not initialized, all axes are contiguous
  if (this->cas_ == nullptr || this->nda_ == 0) {
    for (int i = 0; i < this->nca_; i++) {
      ca[i] = i;
    }
    return;
  }
  // Use exchangeIndex_ to get current state from cas_ array
  // exchangeIndex_ goes from 0 to nda_ as exchanges occur
  std::copy(&(this->cas_[this->exchangeIndex_ * this->nca_]),
            &(this->cas_[this->exchangeIndex_ * this->nca_]) + this->nca_,
            ca);
}

int SlabND::nda() const noexcept {
  return this->nda_;
}

void SlabND::getDa(int* da) const noexcept {
  // Handle single-rank case: das_ not initialized, no distributed axes
  if (this->das_ == nullptr || this->nda_ == 0) {
    return; // Nothing to copy
  }
  // Use exchangeIndex_ to get current state from das_ array
  std::copy(&(this->das_[this->exchangeIndex_ * this->nda_]),
            &(this->das_[this->exchangeIndex_ * this->nda_]) + this->nda_,
            da);
}

void SlabND::prepareForwardExchange() {
  assert((this->exchangeIndex_ >= 0 && this->exchangeIndex_ < this->nda_) &&
         "exchange index out of range");

  this->comm_ = this->comms_[this->nda_ - this->exchangeIndex_ - 1];
  this->subsizeA_ = &(this->subsizes_[this->exchangeIndex_ * this->ndim_]);
  this->subsizeB_ = &(this->subsizes_[(this->exchangeIndex_ + 1) * this->ndim_]);
  this->subarrayA_ = &(this->subarrays_[2 * this->exchangeIndex_ * this->maxCommSize_]);
  this->subarrayB_ = &(this->subarrays_[(2 * this->exchangeIndex_ + 1) * this->maxCommSize_]);

  this->taA_ = this->axesA_[this->exchangeIndex_];
  this->taB_ = this->axesB_[this->exchangeIndex_];

  this->daA_ = &(this->das_[this->exchangeIndex_ * this->nda_]);
  this->caA_ = &(this->cas_[this->exchangeIndex_ * this->nca_]);
  this->daB_ = &(this->das_[(this->exchangeIndex_ + 1) * this->nda_]);
  this->caB_ = &(this->cas_[(this->exchangeIndex_ + 1) * this->nca_]);

  this->exchangeDirection_ = 1;
}

void SlabND::prepareBackwardExchange() {
  // For backward: exchangeIndex_ starts at nda_ and decrements to 1
  // Array accesses use (exchangeIndex_ - 1) which maps to [0, nda_)
  assert((this->exchangeIndex_ > 0 && this->exchangeIndex_ <= this->nda_) &&
         "exchange index out of range");

  this->comm_ = this->comms_[this->nda_ - this->exchangeIndex_];

  this->subsizeA_ = &(this->subsizes_[(this->exchangeIndex_) * this->ndim_]);
  this->subsizeB_ = &(this->subsizes_[(this->exchangeIndex_ - 1) * this->ndim_]);
  this->subarrayA_ = &(this->subarrays_[(2 * (this->exchangeIndex_ - 1) + 1) * this->maxCommSize_]);
  this->subarrayB_ = &(this->subarrays_[2 * (this->exchangeIndex_ - 1) * this->maxCommSize_]);

  this->taA_ = this->axesB_[this->exchangeIndex_ - 1];
  this->taB_ = this->axesA_[this->exchangeIndex_ - 1];

  this->daA_ = &(this->das_[this->exchangeIndex_ * this->nda_]);
  this->caA_ = &(this->cas_[this->exchangeIndex_ * this->nca_]);
  this->daB_ = &(this->das_[(this->exchangeIndex_ - 1) * this->nda_]);
  this->caB_ = &(this->cas_[(this->exchangeIndex_ - 1) * this->nca_]);

  this->exchangeDirection_ = -1;
}

void SlabND::doExchange() {
  if (this->aData_ == nullptr || this->bData_ == nullptr) {
    std::cerr << "Error: Missing buffers." << std::endl;
    return;
  }

#if SHAFFT_BACKEND_HIPFFT && !SHAFFT_GPU_AWARE_MPI
  // Host-staging mode: copy device -> host, MPI exchange, copy host -> device
  {
    size_t allocElems = this->localAllocSize();
    size_t bufBytes = allocElems * this->elemSize_;

    // Lazily allocate host staging buffers
    if (this->hostA_ == nullptr) {
      this->hostA_ = std::malloc(bufBytes);
      if (!this->hostA_) {
        std::cerr << "Error: Failed to allocate host staging buffer A (" << bufBytes << " bytes)"
                  << std::endl;
        return;
      }
    }
    if (this->hostB_ == nullptr) {
      this->hostB_ = std::malloc(bufBytes);
      if (!this->hostB_) {
        std::cerr << "Error: Failed to allocate host staging buffer B (" << bufBytes << " bytes)"
                  << std::endl;
        return;
      }
    }

    // Copy input from device to host
    hipError_t err = hipMemcpy(this->hostA_, this->aData_, bufBytes, hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
      std::cerr << "Error: hipMemcpy D2H failed: " << hipGetErrorString(err) << std::endl;
      return;
    }

    // Do MPI exchange on host buffers
    exchange(this->comm_, this->hostA_, this->subarrayA_, this->hostB_, this->subarrayB_);

    // Copy result back to device
    err = hipMemcpy(this->bData_, this->hostB_, bufBytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
      std::cerr << "Error: hipMemcpy H2D failed: " << hipGetErrorString(err) << std::endl;
      return;
    }
  }
#else
  // GPU-aware MPI (or FFTW backend): exchange directly on buffer pointers
  exchange(this->comm_, this->aData_, this->subarrayA_, this->bData_, this->subarrayB_);
#endif

  this->swapBuffers();

  this->exchangeIndex_ += this->exchangeDirection_;

  this->subsize_ = &(this->subsizes_[this->exchangeIndex_ * this->ndim_]);
  this->offset_ = &(this->offsets_[this->exchangeIndex_ * this->ndim_]);
}

void SlabND::getIndices(int index, int ndim, int* size, int* offset, int* indices) {
  for (int i = ndim - 1; i >= 0; i--) {
    indices[i] = index % size[i] + offset[i];
    index /= size[i];
  }
}

int SlabND::getIndex(int* indices, int ndim, int* size) {
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
