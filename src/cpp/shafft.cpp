#include <shafft/shafft.hpp>

#include "detail/array_utils.hpp"
#include "detail/buffer_utils.hpp"
#include "detail/config_info.hpp"
#include "shafft_internal.hpp"

#include <cstring>
#include <limits>

#include "fft1d_method.hpp"
#include "fftnd_method.hpp"

namespace shafft {

// FFTND class

FFTND::~FFTND() noexcept {
  if (data_) {
    data_->release();
    delete data_;
    data_ = nullptr;
  }
}

void FFTND::release() noexcept {
  if (data_) {
    data_->release();
    delete data_;
    data_ = nullptr;
  }
  state_ = PlanState::UNINITIALIZED;
}

FFTND::FFTND(FFTND&& other) noexcept : data_(other.data_) {
  state_ = other.state_;
  other.data_ = nullptr;
  other.state_ = PlanState::UNINITIALIZED;
}

FFTND& FFTND::operator=(FFTND&& other) noexcept {
  if (this != &other) {
    if (data_) {
      data_->release();
      delete data_;
    }
    data_ = other.data_;
    state_ = other.state_;
    other.data_ = nullptr;
    other.state_ = PlanState::UNINITIALIZED;
  }
  return *this;
}

int FFTND::plan() noexcept {
  try {
    if (state_ != PlanState::CONFIGURED) {
      if (state_ == PlanState::PLANNED) {
        return static_cast<int>(Status::ERR_INVALID_STATE); // already planned
      }
      return static_cast<int>(Status::ERR_PLAN_NOT_INIT); // not configured
    }
    if (!data_) {
      return static_cast<int>(Status::ERR_PLAN_NOT_INIT);
    }

    int rc = detail::planNDCreatePlans(data_);
    if (rc != 0) {
      return rc;
    }
    state_ = PlanState::PLANNED;
    return 0;
  }
  SHAFFT_CATCH_RETURN();
}

int FFTND::setBuffers(complexf* data, complexf* work) noexcept {
  try {
    return detail::setBuffers(data_, data, work);
  }
  SHAFFT_CATCH_RETURN();
}

int FFTND::setBuffers(complexd* data, complexd* work) noexcept {
  try {
    return detail::setBuffers(data_, data, work);
  }
  SHAFFT_CATCH_RETURN();
}

int FFTND::getBuffers(complexf** data, complexf** work) noexcept {
  try {
    return detail::getBuffers(data_, data, work);
  }
  SHAFFT_CATCH_RETURN();
}

int FFTND::getBuffers(complexd** data, complexd** work) noexcept {
  try {
    return detail::getBuffers(data_, data, work);
  }
  SHAFFT_CATCH_RETURN();
}

#if SHAFFT_BACKEND_HIPFFT
int FFTND::setBuffers(hipFloatComplex* data, hipFloatComplex* work) noexcept {
  try {
    return detail::setBuffers(data_, data, work);
  }
  SHAFFT_CATCH_RETURN();
}

int FFTND::setBuffers(hipDoubleComplex* data, hipDoubleComplex* work) noexcept {
  try {
    return detail::setBuffers(data_, data, work);
  }
  SHAFFT_CATCH_RETURN();
}

int FFTND::getBuffers(hipFloatComplex** data, hipFloatComplex** work) noexcept {
  try {
    return detail::getBuffers(data_, data, work);
  }
  SHAFFT_CATCH_RETURN();
}

int FFTND::getBuffers(hipDoubleComplex** data, hipDoubleComplex** work) noexcept {
  try {
    return detail::getBuffers(data_, data, work);
  }
  SHAFFT_CATCH_RETURN();
}

int FFTND::setStream(hipStream_t stream) noexcept {
  try {
    return detail::setStream(data_, stream);
  }
  SHAFFT_CATCH_RETURN();
}
#endif

#if SHAFFT_BACKEND_FFTW
int FFTND::setBuffers(fftw_complex* data, fftw_complex* work) noexcept {
  try {
    return detail::setBuffers(data_, data, work);
  }
  SHAFFT_CATCH_RETURN();
}

int FFTND::setBuffers(fftwf_complex* data, fftwf_complex* work) noexcept {
  try {
    return detail::setBuffers(data_, data, work);
  }
  SHAFFT_CATCH_RETURN();
}

int FFTND::getBuffers(fftw_complex** data, fftw_complex** work) noexcept {
  try {
    return detail::getBuffers(data_, data, work);
  }
  SHAFFT_CATCH_RETURN();
}

int FFTND::getBuffers(fftwf_complex** data, fftwf_complex** work) noexcept {
  try {
    return detail::getBuffers(data_, data, work);
  }
  SHAFFT_CATCH_RETURN();
}
#endif

int FFTND::execute(FFTDirection direction) noexcept {
  try {
    if (state_ != PlanState::PLANNED)
      return static_cast<int>(Status::ERR_PLAN_NOT_INIT);
    return detail::execute(data_, direction);
  }
  SHAFFT_CATCH_RETURN();
}

int FFTND::normalize() noexcept {
  try {
    return detail::normalize(data_);
  }
  SHAFFT_CATCH_RETURN();
}

size_t FFTND::allocSize() const noexcept {
  if (!data_)
    return 0;
  size_t size = 0;
  detail::getAllocSize(data_, &size);
  return size;
}

int FFTND::getLayout(std::vector<size_t>& subsize,
                     std::vector<size_t>& offset,
                     TensorLayout layout) const noexcept {
  try {
    if (!data_ || !data_->slab_)
      return static_cast<int>(Status::ERR_PLAN_NOT_INIT);
    const int n = data_->slab_->ndim();
    subsize.resize(n);
    offset.resize(n);
    return detail::getLayout(data_, subsize.data(), offset.data(), layout);
  }
  SHAFFT_CATCH_RETURN();
}

int FFTND::getAxes(std::vector<int>& ca, std::vector<int>& da, TensorLayout layout) const noexcept {
  try {
    if (!data_ || !data_->slab_)
      return static_cast<int>(Status::ERR_PLAN_NOT_INIT);
    ca.resize(data_->slab_->nca());
    da.resize(data_->slab_->nda());
    return detail::getAxes(data_, ca.data(), da.data(), layout);
  }
  SHAFFT_CATCH_RETURN();
}

bool FFTND::isActive() const noexcept {
  if (!data_ || !data_->slab_)
    return false;
  return data_->slab_->isActive();
}

size_t FFTND::globalSize() const noexcept {
  if (!data_ || !data_->slab_)
    return 0;
  // Compute product of all global dimensions
  const int n = data_->slab_->ndim();
  std::vector<int> size(n);
  data_->slab_->getSize(size.data());
  return detail::prodN<int, size_t>(size.data(), n);
}

int FFTND::ndim() const noexcept {
  try {
    if (!data_ || !data_->slab_)
      return 0;
    return data_->slab_->ndim();
  }
  SHAFFT_CATCH_RETURN();
}

FFTType FFTND::fftType() const noexcept {
  if (!data_)
    return FFTType::C2C; // default
  return data_->fft_type;
}

int FFTND::setBuffersRaw(void* data, void* work) noexcept {
  try {
    if (!data_)
      return static_cast<int>(Status::ERR_NULLPTR);
    if (!data_->slab_)
      return static_cast<int>(Status::ERR_PLAN_NOT_INIT);
    if (!data || !work)
      return static_cast<int>(Status::ERR_NULLPTR);
    data_->slab_->setBuffers(data, work);
    return static_cast<int>(Status::SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

int FFTND::getBuffersRaw(void** data, void** work) noexcept {
  try {
    if (!data_)
      return static_cast<int>(Status::ERR_NULLPTR);
    if (!data_->slab_)
      return static_cast<int>(Status::ERR_PLAN_NOT_INIT);
    if (!data || !work)
      return static_cast<int>(Status::ERR_NULLPTR);
    data_->slab_->getBuffers(data, work);
    return static_cast<int>(Status::SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

// Free functions

#if SHAFFT_BACKEND_HIPFFT
int setStream(detail::PlanBase* plan, hipStream_t stream) {
  try {
    return detail::setStream(plan, stream);
  }
  SHAFFT_CATCH_RETURN();
}

int getBuffers(detail::PlanBase* plan, hipFloatComplex** data, hipFloatComplex** work) noexcept {
  try {
    return detail::getBuffers(plan, reinterpret_cast<void**>(data), reinterpret_cast<void**>(work));
  }
  SHAFFT_CATCH_RETURN();
}

int getBuffers(detail::PlanBase* plan, hipDoubleComplex** data, hipDoubleComplex** work) noexcept {
  try {
    return detail::getBuffers(plan, reinterpret_cast<void**>(data), reinterpret_cast<void**>(work));
  }
  SHAFFT_CATCH_RETURN();
}

int setBuffers(detail::PlanBase* plan, hipFloatComplex* data, hipFloatComplex* work) noexcept {
  try {
    return detail::setBuffers(plan, data, work);
  }
  SHAFFT_CATCH_RETURN();
}
int setBuffers(detail::PlanBase* plan, hipDoubleComplex* data, hipDoubleComplex* work) noexcept {
  try {
    return detail::setBuffers(plan, data, work);
  }
  SHAFFT_CATCH_RETURN();
}
#endif
#if SHAFFT_BACKEND_FFTW
int getBuffers(detail::PlanBase* plan, fftw_complex** data, fftw_complex** work) noexcept {
  try {
    return detail::getBuffers(plan, reinterpret_cast<void**>(data), reinterpret_cast<void**>(work));
  }
  SHAFFT_CATCH_RETURN();
}
int getBuffers(detail::PlanBase* plan, fftwf_complex** data, fftwf_complex** work) noexcept {
  try {
    return detail::getBuffers(plan, reinterpret_cast<void**>(data), reinterpret_cast<void**>(work));
  }
  SHAFFT_CATCH_RETURN();
}
int setBuffers(detail::PlanBase* plan, fftw_complex* data, fftw_complex* work) noexcept {
  try {
    return detail::setBuffers(plan, data, work);
  }
  SHAFFT_CATCH_RETURN();
}
int setBuffers(detail::PlanBase* plan, fftwf_complex* data, fftwf_complex* work) noexcept {
  try {
    return detail::setBuffers(plan, data, work);
  }
  SHAFFT_CATCH_RETURN();
}
#endif

int planNDCreate(detail::FFTNDPlan** out) {
  try {
    if (!out)
      return (int)shafft::Status::ERR_NULLPTR;
    return detail::planCreate(out);
  }
  SHAFFT_CATCH_RETURN();
}

int FFT1DCreate(detail::FFT1DPlan** out) {
  try {
    if (!out)
      return (int)shafft::Status::ERR_NULLPTR;
    *out = new detail::FFT1DPlan{};
    return (int)shafft::Status::SUCCESS;
  }
  SHAFFT_CATCH_RETURN();
}

int configurationND(const std::vector<size_t>& size,
                    FFTType precision,
                    std::vector<int>& commDims,
                    int& nda,
                    std::vector<size_t>& subsize,
                    std::vector<size_t>& offset,
                    int& commSize,
                    DecompositionStrategy strategy,
                    size_t memLimit,
                    MPI_Comm comm) {
  try {
    const int ndim = static_cast<int>(size.size());
    if ((int)subsize.size() != ndim || (int)offset.size() != ndim || (int)commDims.size() != ndim) {
      return (int)shafft::Status::ERR_DIM_MISMATCH;
    }

    // Check for INT_MAX overflow and convert to int for internal API
    std::vector<int> intSize(ndim);
    for (int i = 0; i < ndim; ++i) {
      if (size[i] > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return (int)shafft::Status::ERR_SIZE_OVERFLOW;
      }
      intSize[i] = static_cast<int>(size[i]);
    }

    // Internal API uses int, convert on output
    std::vector<int> intSubsize(ndim), intOffset(ndim);
    int rc = detail::configurationND(ndim,
                                     intSize.data(),
                                     precision,
                                     commDims.data(),
                                     &nda,
                                     intSubsize.data(),
                                     intOffset.data(),
                                     &commSize,
                                     strategy,
                                     memLimit,
                                     comm);
    if (rc == 0) {
      for (int i = 0; i < ndim; ++i) {
        subsize[i] = static_cast<size_t>(intSubsize[i]);
        offset[i] = static_cast<size_t>(intOffset[i]);
      }
    }
    return rc;
  }
  SHAFFT_CATCH_RETURN();
}

int destroy(detail::PlanBase** plan) {
  try {
    if (!plan || !*plan)
      return static_cast<int>(Status::ERR_NULLPTR);
    int rc = detail::destroy(plan);
    if (rc == 0)
      *plan = nullptr;
    return rc;
  }
  SHAFFT_CATCH_RETURN();
}

int getLayout(const detail::PlanBase* plan,
              std::vector<size_t>& subsize,
              std::vector<size_t>& offset,
              shafft::TensorLayout layout) {
  try {
    if (!plan || !plan->slab())
      return static_cast<int>(Status::ERR_PLAN_NOT_INIT);
    const int n = plan->slab()->ndim();
    subsize.resize(n);
    offset.resize(n);
    return detail::getLayout(
        const_cast<detail::PlanBase*>(plan), subsize.data(), offset.data(), layout);
  }
  SHAFFT_CATCH_RETURN();
}

int getAxes(const detail::PlanBase* plan,
            std::vector<int>& ca,
            std::vector<int>& da,
            shafft::TensorLayout layout) {
  try {
    if (!plan || !plan->slab())
      return static_cast<int>(Status::ERR_PLAN_NOT_INIT);
    ca.resize(plan->slab()->nca());
    da.resize(plan->slab()->nda());
    return detail::getAxes(const_cast<detail::PlanBase*>(plan), ca.data(), da.data(), layout);
  }
  SHAFFT_CATCH_RETURN();
}

int getAllocSize(const detail::PlanBase* plan, size_t& localAllocSize) {
  try {
    return detail::getAllocSize(const_cast<detail::PlanBase*>(plan), &localAllocSize);
  }
  SHAFFT_CATCH_RETURN();
}

int execute(detail::PlanBase* plan, shafft::FFTDirection direction) {
  try {
    return detail::execute(plan, direction);
  }
  SHAFFT_CATCH_RETURN();
}

int normalize(detail::PlanBase* plan) {
  try {
    return detail::normalize(plan);
  }
  SHAFFT_CATCH_RETURN();
}

// Backend-agnostic buffer functions

int setBuffers(detail::PlanBase* plan, complexf* data, complexf* work) noexcept {
  try {
    return detail::setBuffers(plan, data, work);
  }
  SHAFFT_CATCH_RETURN();
}

int setBuffers(detail::PlanBase* plan, complexd* data, complexd* work) noexcept {
  try {
    return detail::setBuffers(plan, data, work);
  }
  SHAFFT_CATCH_RETURN();
}

int getBuffers(detail::PlanBase* plan, complexf** data, complexf** work) noexcept {
  try {
    return detail::getBuffers(plan, reinterpret_cast<void**>(data), reinterpret_cast<void**>(work));
  }
  SHAFFT_CATCH_RETURN();
}

int getBuffers(detail::PlanBase* plan, complexd** data, complexd** work) noexcept {
  try {
    return detail::getBuffers(plan, reinterpret_cast<void**>(data), reinterpret_cast<void**>(work));
  }
  SHAFFT_CATCH_RETURN();
}

// Portable memory allocation

int allocBuffer(size_t count, complexf** buf) noexcept {
  return detail::allocBufferT(count, buf);
}

int allocBuffer(size_t count, complexd** buf) noexcept {
  return detail::allocBufferT(count, buf);
}

int freeBuffer(complexf* buf) noexcept {
  return detail::freeBufferT(buf);
}

int freeBuffer(complexd* buf) noexcept {
  return detail::freeBufferT(buf);
}

// Portable memory copy

int copyToBuffer(complexf* dst, const complexf* src, size_t count) noexcept {
  return detail::copyToBufferT(dst, src, count);
}

int copyToBuffer(complexd* dst, const complexd* src, size_t count) noexcept {
  return detail::copyToBufferT(dst, src, count);
}

int copyFromBuffer(complexf* dst, const complexf* src, size_t count) noexcept {
  return detail::copyFromBufferT(dst, src, count);
}

int copyFromBuffer(complexd* dst, const complexd* src, size_t count) noexcept {
  return detail::copyFromBufferT(dst, src, count);
}

// FFT1D class

FFT1D::~FFT1D() noexcept {
  // Inline cleanup to avoid virtual call during destruction
  if (data_) {
    data_->release();
    delete data_;
    data_ = nullptr;
  }
}

void FFT1D::release() noexcept {
  if (data_) {
    data_->release();
    delete data_;
    data_ = nullptr;
  }
  state_ = PlanState::UNINITIALIZED;
}

FFT1D::FFT1D(FFT1D&& other) noexcept : data_(other.data_) {
  state_ = other.state_;
  other.data_ = nullptr;
  other.state_ = PlanState::UNINITIALIZED;
}

FFT1D& FFT1D::operator=(FFT1D&& other) noexcept {
  if (this != &other) {
    release();
    data_ = other.data_;
    state_ = other.state_;
    other.data_ = nullptr;
    other.state_ = PlanState::UNINITIALIZED;
  }
  return *this;
}

int FFT1D::plan() noexcept {
  try {
    if (state_ != PlanState::CONFIGURED) {
      return static_cast<int>(Status::ERR_INVALID_STATE);
    }

    int rc = detail::fft1dCreatePlans(data_);
    if (rc != 0)
      return rc;

    state_ = PlanState::PLANNED;
    return static_cast<int>(Status::SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

int FFT1D::setBuffers(complexf* data, complexf* work) noexcept {
  try {
    shafft::detail::PlanBase* base = data_;
    return detail::setBuffers(base, static_cast<void*>(data), static_cast<void*>(work));
  }
  SHAFFT_CATCH_RETURN();
}

int FFT1D::setBuffers(complexd* data, complexd* work) noexcept {
  try {
    shafft::detail::PlanBase* base = data_;
    return detail::setBuffers(base, static_cast<void*>(data), static_cast<void*>(work));
  }
  SHAFFT_CATCH_RETURN();
}

int FFT1D::getBuffers(complexf** data, complexf** work) noexcept {
  try {
    void *d = nullptr, *w = nullptr;
    shafft::detail::PlanBase* base = data_;
    int rc = detail::getBuffers(base, &d, &w);
    if (rc == 0) {
      if (data)
        *data = static_cast<complexf*>(d);
      if (work)
        *work = static_cast<complexf*>(w);
    }
    return rc;
  }
  SHAFFT_CATCH_RETURN();
}

int FFT1D::getBuffers(complexd** data, complexd** work) noexcept {
  try {
    void *d = nullptr, *w = nullptr;
    shafft::detail::PlanBase* base = data_;
    int rc = detail::getBuffers(base, &d, &w);
    if (rc == 0) {
      if (data)
        *data = static_cast<complexd*>(d);
      if (work)
        *work = static_cast<complexd*>(w);
    }
    return rc;
  }
  SHAFFT_CATCH_RETURN();
}

#if SHAFFT_BACKEND_HIPFFT
int FFT1D::setBuffers(hipFloatComplex* data, hipFloatComplex* work) noexcept {
  try {
    return setBuffers(reinterpret_cast<complexf*>(data), reinterpret_cast<complexf*>(work));
  }
  SHAFFT_CATCH_RETURN();
}

int FFT1D::setBuffers(hipDoubleComplex* data, hipDoubleComplex* work) noexcept {
  try {
    return setBuffers(reinterpret_cast<complexd*>(data), reinterpret_cast<complexd*>(work));
  }
  SHAFFT_CATCH_RETURN();
}

int FFT1D::getBuffers(hipFloatComplex** data, hipFloatComplex** work) noexcept {
  try {
    return getBuffers(reinterpret_cast<complexf**>(data), reinterpret_cast<complexf**>(work));
  }
  SHAFFT_CATCH_RETURN();
}

int FFT1D::getBuffers(hipDoubleComplex** data, hipDoubleComplex** work) noexcept {
  try {
    return getBuffers(reinterpret_cast<complexd**>(data), reinterpret_cast<complexd**>(work));
  }
  SHAFFT_CATCH_RETURN();
}

int FFT1D::setStream(hipStream_t stream) noexcept {
  try {
    shafft::detail::PlanBase* base = data_;
    return detail::setStream(base, stream);
  }
  SHAFFT_CATCH_RETURN();
}
#endif

#if SHAFFT_BACKEND_FFTW
int FFT1D::setBuffers(fftw_complex* data, fftw_complex* work) noexcept {
  try {
    return setBuffers(reinterpret_cast<complexd*>(data), reinterpret_cast<complexd*>(work));
  }
  SHAFFT_CATCH_RETURN();
}

int FFT1D::setBuffers(fftwf_complex* data, fftwf_complex* work) noexcept {
  try {
    return setBuffers(reinterpret_cast<complexf*>(data), reinterpret_cast<complexf*>(work));
  }
  SHAFFT_CATCH_RETURN();
}

int FFT1D::getBuffers(fftw_complex** data, fftw_complex** work) noexcept {
  try {
    return getBuffers(reinterpret_cast<complexd**>(data), reinterpret_cast<complexd**>(work));
  }
  SHAFFT_CATCH_RETURN();
}

int FFT1D::getBuffers(fftwf_complex** data, fftwf_complex** work) noexcept {
  try {
    return getBuffers(reinterpret_cast<complexf**>(data), reinterpret_cast<complexf**>(work));
  }
  SHAFFT_CATCH_RETURN();
}
#endif

int FFT1D::execute(FFTDirection direction) noexcept {
  try {
    if (state_ != PlanState::PLANNED)
      return static_cast<int>(Status::ERR_PLAN_NOT_INIT);
    shafft::detail::PlanBase* base = data_;
    return detail::execute(base, direction);
  }
  SHAFFT_CATCH_RETURN();
}

int FFT1D::normalize() noexcept {
  try {
    shafft::detail::PlanBase* base = data_;
    return detail::normalize(base);
  }
  SHAFFT_CATCH_RETURN();
}

std::vector<size_t> FFT1D::globalShape() const noexcept {
  size_t globalN = 0;
  auto* base = const_cast<shafft::detail::FFT1DPlan*>(data_);
  if (detail::getGlobalSize(base, &globalN) != 0)
    return {};
  return {globalN};
}

size_t FFT1D::globalSize() const noexcept {
  size_t globalN = 0;
  auto* base = const_cast<shafft::detail::FFT1DPlan*>(data_);
  if (detail::getGlobalSize(base, &globalN) != 0)
    return 0;
  return globalN;
}

size_t FFT1D::localSize() const noexcept {
  size_t localN = 0, localStart = 0;
  auto* base = const_cast<shafft::detail::FFT1DPlan*>(data_);
  if (detail::getLayout(base, &localN, &localStart, TensorLayout::INITIAL) != 0)
    return 0;
  return localN;
}

size_t FFT1D::allocSize() const noexcept {
  size_t localAllocSize = 0;
  auto* base = const_cast<shafft::detail::FFT1DPlan*>(data_);
  if (detail::getAllocSize(base, &localAllocSize) != 0)
    return 0;
  return localAllocSize;
}

int FFT1D::getLayout(std::vector<size_t>& localShape,
                     std::vector<size_t>& offset,
                     TensorLayout layout) const noexcept {
  try {
    size_t localN = 0, localStart = 0;
    auto* base = const_cast<shafft::detail::FFT1DPlan*>(data_);
    int rc = detail::getLayout(base, &localN, &localStart, layout);
    if (rc != 0)
      return rc;
    localShape = {localN};
    offset = {localStart};
    return static_cast<int>(Status::SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

int FFT1D::getAxes(std::vector<int>& ca, std::vector<int>& da, TensorLayout layout) const noexcept {
  try {
    if (!data_ || !data_->slab())
      return static_cast<int>(Status::ERR_PLAN_NOT_INIT);
    ca.resize(data_->slab()->nca()); // 0 for 1D
    da.resize(data_->slab()->nda()); // 1 for 1D
    return detail::getAxes(
        const_cast<shafft::detail::FFT1DPlan*>(data_), ca.data(), da.data(), layout);
  }
  SHAFFT_CATCH_RETURN();
}

bool FFT1D::isActive() const noexcept {
  int isActive = 0;
  auto* base = const_cast<shafft::detail::FFT1DPlan*>(data_);
  detail::isActive(base, &isActive);
  return isActive != 0;
}

int FFT1D::ndim() const noexcept {
  try {
    return 1; // Always 1 for FFT1D
  }
  SHAFFT_CATCH_RETURN();
}

FFTType FFT1D::fftType() const noexcept {
  FFTType precision = FFTType::C2C;
  if (data_) {
    auto* base = const_cast<shafft::detail::FFT1DPlan*>(data_);
    detail::getPrecision(base, &precision);
  }
  return precision;
}

int FFT1D::setBuffersRaw(void* data, void* work) noexcept {
  try {
    shafft::detail::PlanBase* base = data_;
    return detail::setBuffers(base, data, work);
  }
  SHAFFT_CATCH_RETURN();
}

int FFT1D::getBuffersRaw(void** data, void** work) noexcept {
  try {
    shafft::detail::PlanBase* base = data_;
    return detail::getBuffers(base, data, work);
  }
  SHAFFT_CATCH_RETURN();
}

int configuration1D(
    size_t globalN, size_t& localN, size_t& localStart, FFTType precision, MPI_Comm comm) {
  try {
    size_t localAllocSize = 0;
    return detail::configuration1D(globalN, &localN, &localStart, &localAllocSize, precision, comm);
  }
  SHAFFT_CATCH_RETURN();
}

int planND(detail::FFTNDPlan* plan,
           const std::vector<int>& commDims,
           const std::vector<size_t>& dimensions,
           FFTType precision,
           MPI_Comm comm) {
  try {
    if (!plan) {
      return (int)shafft::Status::ERR_NULLPTR;
    }
    const int ndim = static_cast<int>(commDims.size());
    if (ndim != static_cast<int>(dimensions.size())) {
      return (int)shafft::Status::ERR_DIM_MISMATCH;
    }

    // Check for INT_MAX overflow and convert to int for internal API
    std::vector<int> intDimensions(ndim);
    for (int i = 0; i < ndim; ++i) {
      if (dimensions[i] > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return (int)shafft::Status::ERR_SIZE_OVERFLOW;
      }
      intDimensions[i] = static_cast<int>(dimensions[i]);
    }

    int rc = detail::planNDConfigure(
        plan, ndim, const_cast<int*>(commDims.data()), intDimensions.data(), precision, comm);
    if (rc != 0)
      return rc;
    return detail::planNDCreatePlans(plan);
  }
  SHAFFT_CATCH_RETURN();
}

int finalize() noexcept {
  int rc1 = fftndFinalize();
  int rc2 = fft1dFinalize();
  // Return first error if any
  return (rc1 != 0) ? rc1 : rc2;
}

// ---- getCommunicator implementations --------------------------------------

int FFTND::getCommunicator(MPI_Comm* outComm) const noexcept {
  try {
    if (!outComm)
      return static_cast<int>(Status::ERR_NULLPTR);
    if (!data_)
      return static_cast<int>(Status::ERR_PLAN_NOT_INIT);
    MPI_Comm src = data_->getComm();
    if (src == MPI_COMM_NULL) {
      *outComm = MPI_COMM_NULL;
      return 0;
    }
    int rc = MPI_Comm_dup(src, outComm);
    return (rc == MPI_SUCCESS) ? 0 : static_cast<int>(Status::ERR_MPI);
  }
  SHAFFT_CATCH_RETURN();
}

int FFT1D::getCommunicator(MPI_Comm* outComm) const noexcept {
  try {
    if (!outComm)
      return static_cast<int>(Status::ERR_NULLPTR);
    if (!data_)
      return static_cast<int>(Status::ERR_PLAN_NOT_INIT);
    MPI_Comm src = data_->getComm();
    if (src == MPI_COMM_NULL) {
      *outComm = MPI_COMM_NULL;
      return 0;
    }
    int rc = MPI_Comm_dup(src, outComm);
    return (rc == MPI_SUCCESS) ? 0 : static_cast<int>(Status::ERR_MPI);
  }
  SHAFFT_CATCH_RETURN();
}

// ---- Direct-init overloads (legacy, non-config) ---------------------------

int FFTND::init(const std::vector<int>& commDims,
                const std::vector<size_t>& dimensions,
                FFTType type,
                MPI_Comm comm,
                TransformLayout output) noexcept {
  try {
    if (state_ != PlanState::UNINITIALIZED) {
      return static_cast<int>(Status::ERR_INVALID_STATE); // already initialized
    }
    if (data_) {
      return static_cast<int>(Status::ERR_INVALID_STATE); // already has data
    }
    int rc = detail::planCreate(&data_);
    if (rc != 0)
      return rc;

    const int ndim = static_cast<int>(commDims.size());
    if (ndim != static_cast<int>(dimensions.size())) {
      delete data_;
      data_ = nullptr;
      return static_cast<int>(Status::ERR_DIM_MISMATCH);
    }

    // Check for INT_MAX overflow and convert to int for internal API
    std::vector<int> intDimensions(ndim);
    for (int i = 0; i < ndim; ++i) {
      if (dimensions[i] > static_cast<size_t>(std::numeric_limits<int>::max())) {
        delete data_;
        data_ = nullptr;
        return static_cast<int>(Status::ERR_SIZE_OVERFLOW);
      }
      intDimensions[i] = static_cast<int>(dimensions[i]);
    }

    // Configure only - defer backend plan creation to plan()
    rc = detail::planNDConfigure(
        data_, ndim, const_cast<int*>(commDims.data()), intDimensions.data(), type, comm);
    if (rc != 0) {
      delete data_;
      data_ = nullptr;
      return rc;
    }
    data_->setOutputLayoutPolicy(output);
    state_ = PlanState::CONFIGURED;
    return 0;
  }
  SHAFFT_CATCH_RETURN();
}

int FFT1D::init(
    size_t globalN, size_t localN, size_t localStart, FFTType precision, MPI_Comm comm) noexcept {
  try {
    if (state_ != PlanState::UNINITIALIZED)
      return static_cast<int>(Status::ERR_INVALID_STATE); // Already initialized

    int rc = detail::fft1dCreate(&data_);
    if (rc != 0)
      return rc;

    rc = detail::fft1dConfigure(data_, globalN, localN, localStart, precision, comm);
    if (rc != 0) {
      delete data_;
      data_ = nullptr;
      return rc;
    }

    state_ = PlanState::CONFIGURED;
    return static_cast<int>(Status::SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

// ---- init-from-config overloads -------------------------------------------

int FFTND::init(shafft_nd_config_t& cfg) noexcept {
  try {
    if (state_ != PlanState::UNINITIALIZED)
      return static_cast<int>(Status::ERR_INVALID_STATE);

    // Auto-resolve if needed
    if (!(cfg.flags & SHAFFT_CONFIG_RESOLVED)) {
      int rc = detail::configNDResolve(&cfg);
      if (rc != 0)
        return rc;
    }

    // Use the active subcommunicator created by resolve.
    // Inactive ranks have MPI_COMM_NULL, which planNDConfigure handles.
    MPI_Comm comm = cfg.activeComm;

    int rc = detail::planCreate(&data_);
    if (rc != 0)
      return rc;

    std::vector<int> intShape(cfg.ndim);
    for (int i = 0; i < cfg.ndim; ++i)
      intShape[i] = static_cast<int>(cfg.globalShape[i]);

    rc = detail::planNDConfigure(data_,
                                 cfg.ndim,
                                 cfg.commDims,
                                 intShape.data(),
                                 (cfg.precision == SHAFFT_Z2Z) ? FFTType::Z2Z : FFTType::C2C,
                                 comm);
    if (rc != 0) {
      delete data_;
      data_ = nullptr;
      return rc;
    }

    // Read output policy from config struct
    TransformLayout output = (cfg.outputPolicy == SHAFFT_LAYOUT_INITIAL)
                                 ? TransformLayout::INITIAL
                                 : TransformLayout::REDISTRIBUTED;
    data_->setOutputLayoutPolicy(output);
    state_ = PlanState::CONFIGURED;
    return 0;
  }
  SHAFFT_CATCH_RETURN();
}

int FFT1D::init(shafft_1d_config_t& cfg) noexcept {
  try {
    if (state_ != PlanState::UNINITIALIZED)
      return static_cast<int>(Status::ERR_INVALID_STATE);

    // Auto-resolve if needed
    if (!(cfg.flags & SHAFFT_CONFIG_RESOLVED)) {
      int rc = detail::config1DResolve(&cfg);
      if (rc != 0)
        return rc;
    }

    // Use the active subcommunicator created by resolve.
    // Inactive ranks have MPI_COMM_NULL, which fft1dConfigure handles.
    MPI_Comm comm = cfg.activeComm;

    int rc = detail::fft1dCreate(&data_);
    if (rc != 0)
      return rc;

    FFTType prec = (cfg.precision == SHAFFT_Z2Z) ? FFTType::Z2Z : FFTType::C2C;
    rc = detail::fft1dConfigure(
        data_, cfg.globalSize, cfg.initial.localSize, cfg.initial.localStart, prec, comm);
    if (rc != 0) {
      delete data_;
      data_ = nullptr;
      return rc;
    }
    state_ = PlanState::CONFIGURED;
    return 0;
  }
  SHAFFT_CATCH_RETURN();
}

// ---- ConfigND RAII wrapper -------------------------------------------------

ConfigND::ConfigND(const std::vector<size_t>& globalShape,
                   FFTType precision,
                   const std::vector<int>& commDims,
                   int hintNda,
                   DecompositionStrategy strategy,
                   TransformLayout outputPolicy,
                   size_t memLimit,
                   MPI_Comm comm) noexcept {
  int ndim = static_cast<int>(globalShape.size());
  shafft_t p = (precision == FFTType::Z2Z) ? SHAFFT_Z2Z : SHAFFT_C2C;
  shafft_decomposition_strategy_t s =
      (strategy == DecompositionStrategy::MINIMIZE_NDA) ? SHAFFT_MINIMIZE_NDA : SHAFFT_MAXIMIZE_NDA;
  shafft_transform_layout_t op = (outputPolicy == TransformLayout::INITIAL)
                                     ? SHAFFT_LAYOUT_INITIAL
                                     : SHAFFT_LAYOUT_REDISTRIBUTED;
  const int* cd = commDims.empty() ? nullptr : commDims.data();
  status_ =
      detail::configNDInit(&cfg_, ndim, globalShape.data(), p, cd, hintNda, s, op, memLimit, comm);
}

ConfigND::~ConfigND() noexcept {
  detail::configNDRelease(&cfg_);
}

ConfigND::ConfigND(ConfigND&& other) noexcept : cfg_(other.cfg_), status_(other.status_) {
  std::memset(&other.cfg_, 0, sizeof(other.cfg_));
  other.status_ = -1;
}

ConfigND& ConfigND::operator=(ConfigND&& other) noexcept {
  if (this != &other) {
    detail::configNDRelease(&cfg_);
    cfg_ = other.cfg_;
    status_ = other.status_;
    std::memset(&other.cfg_, 0, sizeof(other.cfg_));
    other.status_ = -1;
  }
  return *this;
}

int ConfigND::resolve() noexcept {
  if (status_ != 0)
    return status_;
  return detail::configNDResolve(&cfg_);
}

// ---- Config1D RAII wrapper -------------------------------------------------

Config1D::Config1D(size_t globalSize, FFTType precision, MPI_Comm comm) noexcept {
  shafft_t p = (precision == FFTType::Z2Z) ? SHAFFT_Z2Z : SHAFFT_C2C;
  status_ = detail::config1DInit(&cfg_, globalSize, p, comm);
}

Config1D::~Config1D() noexcept {
  detail::config1DRelease(&cfg_);
}

Config1D::Config1D(Config1D&& other) noexcept : cfg_(other.cfg_), status_(other.status_) {
  std::memset(&other.cfg_, 0, sizeof(other.cfg_));
  other.status_ = -1;
}

Config1D& Config1D::operator=(Config1D&& other) noexcept {
  if (this != &other) {
    detail::config1DRelease(&cfg_);
    cfg_ = other.cfg_;
    status_ = other.status_;
    std::memset(&other.cfg_, 0, sizeof(other.cfg_));
    other.status_ = -1;
  }
  return *this;
}

int Config1D::resolve() noexcept {
  if (status_ != 0)
    return status_;
  return detail::config1DResolve(&cfg_);
}

} // namespace shafft
