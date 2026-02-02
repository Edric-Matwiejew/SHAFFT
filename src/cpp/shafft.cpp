#include <shafft/shafft.hpp>

#include "_shafft.hpp"

#include <cstdlib>  // std::malloc, std::free
#include <cstring>  // std::memcpy

using namespace _shafft;

namespace shafft {

//==============================================================================
// Plan class implementation (RAII wrapper)
//==============================================================================

Plan::~Plan() noexcept {
  if (data_) {
    _shafft::destroy(&data_);
  }
}

void Plan::release() noexcept {
  if (data_) {
    _shafft::destroy(&data_);
  }
}

Plan::Plan(Plan&& other) noexcept : data_(other.data_) {
  other.data_ = nullptr;
}

Plan& Plan::operator=(Plan&& other) noexcept {
  if (this != &other) {
    if (data_) {
      _shafft::destroy(&data_);
    }
    data_ = other.data_;
    other.data_ = nullptr;
  }
  return *this;
}

int Plan::init(int nda, const std::vector<int>& dimensions, FFTType type, MPI_Comm comm) noexcept {
  if (data_) {
    return static_cast<int>(Status::SHAFFT_ERR_PLAN_NOT_INIT);  // already initialized
  }
  int rc = _shafft::planCreate(&data_);
  if (rc != 0)
    return rc;

  const int ndim = static_cast<int>(dimensions.size());
  rc = _shafft::planNDA(data_, ndim, nda, const_cast<int*>(dimensions.data()), type, comm);
  if (rc != 0) {
    _shafft::destroy(&data_);
    return rc;
  }
  return 0;
}

int Plan::initCart(const std::vector<int>& commDims, const std::vector<int>& dimensions,
                   FFTType type, MPI_Comm comm) noexcept {
  if (data_) {
    return static_cast<int>(Status::SHAFFT_ERR_PLAN_NOT_INIT);
  }
  int rc = _shafft::planCreate(&data_);
  if (rc != 0)
    return rc;

  const int ndim = static_cast<int>(commDims.size());
  if (ndim != static_cast<int>(dimensions.size())) {
    _shafft::destroy(&data_);
    return static_cast<int>(Status::SHAFFT_ERR_DIM_MISMATCH);
  }

  rc = _shafft::planCart(data_, ndim, const_cast<int*>(commDims.data()),
                         const_cast<int*>(dimensions.data()), type, comm);
  if (rc != 0) {
    _shafft::destroy(&data_);
    return rc;
  }
  return 0;
}

int Plan::setBuffers(complexf* data, complexf* work) noexcept {
  return _shafft::setBuffers(data_, data, work);
}

int Plan::setBuffers(complexd* data, complexd* work) noexcept {
  return _shafft::setBuffers(data_, data, work);
}

int Plan::getBuffers(complexf** data, complexf** work) noexcept {
  return _shafft::getBuffers(data_, data, work);
}

int Plan::getBuffers(complexd** data, complexd** work) noexcept {
  return _shafft::getBuffers(data_, data, work);
}

#if SHAFFT_BACKEND_HIPFFT
int Plan::setBuffers(hipFloatComplex* data, hipFloatComplex* work) noexcept {
  return _shafft::setBuffers(data_, data, work);
}

int Plan::setBuffers(hipDoubleComplex* data, hipDoubleComplex* work) noexcept {
  return _shafft::setBuffers(data_, data, work);
}

int Plan::getBuffers(hipFloatComplex** data, hipFloatComplex** work) noexcept {
  return _shafft::getBuffers(data_, data, work);
}

int Plan::getBuffers(hipDoubleComplex** data, hipDoubleComplex** work) noexcept {
  return _shafft::getBuffers(data_, data, work);
}

int Plan::setStream(hipStream_t stream) noexcept {
  return _shafft::setStream(data_, stream);
}
#endif

#if SHAFFT_BACKEND_FFTW
int Plan::setBuffers(fftw_complex* data, fftw_complex* work) noexcept {
  return _shafft::setBuffers(data_, data, work);
}

int Plan::setBuffers(fftwf_complex* data, fftwf_complex* work) noexcept {
  return _shafft::setBuffers(data_, data, work);
}

int Plan::getBuffers(fftw_complex** data, fftw_complex** work) noexcept {
  return _shafft::getBuffers(data_, data, work);
}

int Plan::getBuffers(fftwf_complex** data, fftwf_complex** work) noexcept {
  return _shafft::getBuffers(data_, data, work);
}
#endif

int Plan::execute(FFTDirection direction) noexcept {
  return _shafft::execute(data_, direction);
}

int Plan::normalize() noexcept {
  return _shafft::normalize(data_);
}

size_t Plan::allocSize() const noexcept {
  if (!data_)
    return 0;
  size_t size = 0;
  _shafft::getAllocSize(const_cast<PlanData*>(data_), &size);
  return size;
}

int Plan::getLayout(std::vector<int>& subsize, std::vector<int>& offset,
                    TensorLayout layout) const noexcept {
  return _shafft::getLayout(const_cast<PlanData*>(data_), subsize.data(), offset.data(), layout);
}

int Plan::getAxes(std::vector<int>& ca, std::vector<int>& da, TensorLayout layout) const noexcept {
  return _shafft::getAxes(const_cast<PlanData*>(data_), ca.data(), da.data(), layout);
}

bool Plan::isActive() const noexcept {
  if (!data_ || !data_->slab)
    return false;
  return data_->slab->is_active();
}

//==============================================================================
// Free functions (C-style API)
//==============================================================================

#if SHAFFT_BACKEND_HIPFFT
int setStream(PlanData* plan, hipStream_t stream) {
  return _shafft::setStream(plan, stream);
}

int getBuffers(PlanData* plan, hipFloatComplex** data, hipFloatComplex** work) noexcept {
  return _shafft::getBuffers(plan, data, work);
}

int getBuffers(PlanData* plan, hipDoubleComplex** data, hipDoubleComplex** work) noexcept {
  return _shafft::getBuffers(plan, data, work);
}

int setBuffers(PlanData* plan, hipFloatComplex* data, hipFloatComplex* work) noexcept {
  return _shafft::setBuffers(plan, data, work);
}
int setBuffers(PlanData* plan, hipDoubleComplex* data, hipDoubleComplex* work) noexcept {
  return _shafft::setBuffers(plan, data, work);
}
#endif
#if SHAFFT_BACKEND_FFTW
int getBuffers(PlanData* plan, fftw_complex** data, fftw_complex** work) noexcept {
  return _shafft::getBuffers(plan, data, work);
}
int getBuffers(PlanData* plan, fftwf_complex** data, fftwf_complex** work) noexcept {
  return _shafft::getBuffers(plan, data, work);
}
int setBuffers(PlanData* plan, fftw_complex* data, fftw_complex* work) noexcept {
  return _shafft::setBuffers(plan, data, work);
}
int setBuffers(PlanData* plan, fftwf_complex* data, fftwf_complex* work) noexcept {
  return _shafft::setBuffers(plan, data, work);
}
#endif

int planCreate(PlanData** out) {
  if (!out)
    return (int)shafft::Status::SHAFFT_ERR_NULLPTR;
  return _shafft::planCreate(out);
}

int configurationNDA(const std::vector<int>& size, int& nda, std::vector<int>& subsize,
                     std::vector<int>& offset, std::vector<int>& COMM_DIMS,
                     shafft::FFTType precision, size_t mem_limit, MPI_Comm COMM) {
  const int ndim = static_cast<int>(size.size());
  if ((int)subsize.size() != ndim || (int)offset.size() != ndim || (int)COMM_DIMS.size() != ndim) {
    return (int)shafft::Status::SHAFFT_ERR_DIM_MISMATCH;
  }
  return _shafft::configurationNDA(ndim, const_cast<int*>(size.data()), &nda, subsize.data(),
                                   offset.data(), COMM_DIMS.data(), precision, mem_limit, COMM);
}

int configurationCart(const std::vector<int>& size, std::vector<int>& subsize,
                      std::vector<int>& offset, std::vector<int>& COMM_DIMS, int& COMM_SIZE,
                      shafft::FFTType precision, size_t mem_limit, MPI_Comm COMM) {
  const int ndim = static_cast<int>(size.size());
  if ((int)subsize.size() != ndim || (int)offset.size() != ndim || (int)COMM_DIMS.size() != ndim) {
    return (int)shafft::Status::SHAFFT_ERR_DIM_MISMATCH;
  }
  return _shafft::configurationCart(ndim, const_cast<int*>(size.data()), subsize.data(),
                                    offset.data(), COMM_DIMS.data(), &COMM_SIZE, precision,
                                    mem_limit, COMM);
}

int planNDA(PlanData* plan, int nda, const std::vector<int>& dimensions, shafft::FFTType precision,
            MPI_Comm COMM) {
  const int ndim = static_cast<int>(dimensions.size());
  return _shafft::planNDA(plan, ndim, nda, const_cast<int*>(dimensions.data()), precision, COMM);
}

int planCart(PlanData* plan, const std::vector<int>& COMM_DIMS, const std::vector<int>& dimensions,
             shafft::FFTType precision, MPI_Comm COMM) {
  const int ndim = static_cast<int>(COMM_DIMS.size());
  if (ndim != (int)dimensions.size()) {
    return (int)shafft::Status::SHAFFT_ERR_DIM_MISMATCH;
  }
  return _shafft::planCart(plan, ndim, const_cast<int*>(COMM_DIMS.data()),
                           const_cast<int*>(dimensions.data()), precision, COMM);
}

int destroy(PlanData** plan) {
  return _shafft::destroy(plan);
}

int getLayout(const PlanData* plan, std::vector<int>& subsize, std::vector<int>& offset,
              shafft::TensorLayout layout) {
  return _shafft::getLayout(const_cast<PlanData*>(plan), subsize.data(), offset.data(), layout);
}

int getAxes(const PlanData* plan, std::vector<int>& ca, std::vector<int>& da,
            shafft::TensorLayout layout) {
  return _shafft::getAxes(const_cast<PlanData*>(plan), ca.data(), da.data(), layout);
}

int getAllocSize(const PlanData* plan, size_t& alloc_size) {
  return _shafft::getAllocSize(const_cast<PlanData*>(plan), &alloc_size);
}

int execute(PlanData* plan, shafft::FFTDirection direction) {
  return _shafft::execute(plan, direction);
}

int normalize(PlanData* plan) {
  return _shafft::normalize(plan);
}

//------------------------------------------------------------------------------
// Backend-agnostic buffer functions using std::complex<T>
//------------------------------------------------------------------------------

int setBuffers(PlanData* plan, complexf* data, complexf* work) noexcept {
  return _shafft::setBuffers(plan, data, work);
}

int setBuffers(PlanData* plan, complexd* data, complexd* work) noexcept {
  return _shafft::setBuffers(plan, data, work);
}

int getBuffers(PlanData* plan, complexf** data, complexf** work) noexcept {
  return _shafft::getBuffers(plan, data, work);
}

int getBuffers(PlanData* plan, complexd** data, complexd** work) noexcept {
  return _shafft::getBuffers(plan, data, work);
}

//------------------------------------------------------------------------------
// Portable memory allocation helpers
//------------------------------------------------------------------------------

int allocBuffer(size_t count, complexf** buf) noexcept {
  if (!buf)
    return static_cast<int>(Status::SHAFFT_ERR_NULLPTR);
  if (count == 0) {
    *buf = nullptr;
    return static_cast<int>(Status::SHAFFT_SUCCESS);
  }
#if SHAFFT_BACKEND_HIPFFT
  hipError_t err = hipMalloc(reinterpret_cast<void**>(buf), count * sizeof(complexf));
  if (err != hipSuccess) {
    *buf = nullptr;
    return static_cast<int>(Status::SHAFFT_ERR_ALLOC);
  }
#else
  *buf = static_cast<complexf*>(std::malloc(count * sizeof(complexf)));
  if (!*buf)
    return static_cast<int>(Status::SHAFFT_ERR_ALLOC);
#endif
  return static_cast<int>(Status::SHAFFT_SUCCESS);
}

int allocBuffer(size_t count, complexd** buf) noexcept {
  if (!buf)
    return static_cast<int>(Status::SHAFFT_ERR_NULLPTR);
  if (count == 0) {
    *buf = nullptr;
    return static_cast<int>(Status::SHAFFT_SUCCESS);
  }
#if SHAFFT_BACKEND_HIPFFT
  hipError_t err = hipMalloc(reinterpret_cast<void**>(buf), count * sizeof(complexd));
  if (err != hipSuccess) {
    *buf = nullptr;
    return static_cast<int>(Status::SHAFFT_ERR_ALLOC);
  }
#else
  *buf = static_cast<complexd*>(std::malloc(count * sizeof(complexd)));
  if (!*buf)
    return static_cast<int>(Status::SHAFFT_ERR_ALLOC);
#endif
  return static_cast<int>(Status::SHAFFT_SUCCESS);
}

int freeBuffer(complexf* buf) noexcept {
  if (!buf)
    return static_cast<int>(Status::SHAFFT_SUCCESS);
#if SHAFFT_BACKEND_HIPFFT
  hipError_t err = hipFree(buf);
  if (err != hipSuccess)
    return static_cast<int>(Status::SHAFFT_ERR_BACKEND);
#else
  std::free(buf);
#endif
  return static_cast<int>(Status::SHAFFT_SUCCESS);
}

int freeBuffer(complexd* buf) noexcept {
  if (!buf)
    return static_cast<int>(Status::SHAFFT_SUCCESS);
#if SHAFFT_BACKEND_HIPFFT
  hipError_t err = hipFree(buf);
  if (err != hipSuccess)
    return static_cast<int>(Status::SHAFFT_ERR_BACKEND);
#else
  std::free(buf);
#endif
  return static_cast<int>(Status::SHAFFT_SUCCESS);
}

//------------------------------------------------------------------------------
// Portable memory copy helpers
//------------------------------------------------------------------------------

int copyToBuffer(complexf* dst, const complexf* src, size_t count) noexcept {
  if (!dst || !src)
    return static_cast<int>(Status::SHAFFT_ERR_NULLPTR);
  if (count == 0)
    return static_cast<int>(Status::SHAFFT_SUCCESS);
#if SHAFFT_BACKEND_HIPFFT
  hipError_t err = hipMemcpy(dst, src, count * sizeof(complexf), hipMemcpyHostToDevice);
  if (err != hipSuccess)
    return static_cast<int>(Status::SHAFFT_ERR_BACKEND);
#else
  std::memcpy(dst, src, count * sizeof(complexf));
#endif
  return static_cast<int>(Status::SHAFFT_SUCCESS);
}

int copyToBuffer(complexd* dst, const complexd* src, size_t count) noexcept {
  if (!dst || !src)
    return static_cast<int>(Status::SHAFFT_ERR_NULLPTR);
  if (count == 0)
    return static_cast<int>(Status::SHAFFT_SUCCESS);
#if SHAFFT_BACKEND_HIPFFT
  hipError_t err = hipMemcpy(dst, src, count * sizeof(complexd), hipMemcpyHostToDevice);
  if (err != hipSuccess)
    return static_cast<int>(Status::SHAFFT_ERR_BACKEND);
#else
  std::memcpy(dst, src, count * sizeof(complexd));
#endif
  return static_cast<int>(Status::SHAFFT_SUCCESS);
}

int copyFromBuffer(complexf* dst, const complexf* src, size_t count) noexcept {
  if (!dst || !src)
    return static_cast<int>(Status::SHAFFT_ERR_NULLPTR);
  if (count == 0)
    return static_cast<int>(Status::SHAFFT_SUCCESS);
#if SHAFFT_BACKEND_HIPFFT
  hipError_t err = hipMemcpy(dst, src, count * sizeof(complexf), hipMemcpyDeviceToHost);
  if (err != hipSuccess)
    return static_cast<int>(Status::SHAFFT_ERR_BACKEND);
#else
  std::memcpy(dst, src, count * sizeof(complexf));
#endif
  return static_cast<int>(Status::SHAFFT_SUCCESS);
}

int copyFromBuffer(complexd* dst, const complexd* src, size_t count) noexcept {
  if (!dst || !src)
    return static_cast<int>(Status::SHAFFT_ERR_NULLPTR);
  if (count == 0)
    return static_cast<int>(Status::SHAFFT_SUCCESS);
#if SHAFFT_BACKEND_HIPFFT
  hipError_t err = hipMemcpy(dst, src, count * sizeof(complexd), hipMemcpyDeviceToHost);
  if (err != hipSuccess)
    return static_cast<int>(Status::SHAFFT_ERR_BACKEND);
#else
  std::memcpy(dst, src, count * sizeof(complexd));
#endif
  return static_cast<int>(Status::SHAFFT_SUCCESS);
}

}  // namespace shafft
