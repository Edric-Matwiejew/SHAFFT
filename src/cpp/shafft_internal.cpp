#include "shafft_internal.hpp"

#include "detail/array_utils.hpp"
#include "detail/error_macros.hpp"
#include "detail/plan_data_base.hpp"
#include "detail/slab_1d.hpp"

#include "common/normalize.hpp"
#include "fft1d_method.hpp"
#include "fftnd_method.hpp"

#include <cmath>
#include <cstdio> // for fprintf
#include <new>

static void getTransformAxes(int* coordSpaces, int nca, int* ca, int* nta, int* ta) {
  *nta = 0;
  for (int i = 0; i < nca; i++) {
    if (coordSpaces[ca[i]] == 0) {
      ta[*nta] = ca[i];
      coordSpaces[ca[i]] = 1;
      *nta += 1;
    }
  }
}

// FFTNDPlan virtual method implementations (declared in plan_nd_data.hpp)

shafft::detail::SlabBase* shafft::detail::FFTNDPlan::slab() noexcept {
  return slab_; // SlabND inherits from SlabBase
}

const shafft::detail::SlabBase* shafft::detail::FFTNDPlan::slab() const noexcept {
  return slab_;
}

void shafft::detail::FFTNDPlan::release() noexcept {
  if (subplans) {
    for (int i = 0; i < nsubplans; ++i) {
      fftndDestroy(subplans[i]);
    }
    delete[] subplans;
    subplans = nullptr;
    nsubplans = 0;
  }

  delete slab_;
  slab_ = nullptr;

  freeActiveComm();

  // Free the plan's owned communicator (dup'd during configure).
  if (comm != MPI_COMM_NULL) {
    MPI_Comm_free(&comm);
    comm = MPI_COMM_NULL;
  }

  norm_exponent_ = 0;
  norm_denominator = 1.0L;
}

int shafft::detail::FFTNDPlan::executeImpl(shafft::FFTDirection direction) noexcept {
  try {
    if (!slab_)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);

    if (!subplans)
      SHAFFT_FAIL(SHAFFT_ERR_PLAN_NOT_INIT);

    const int nsp = nsubplans;
    if (nsp < 1)
      SHAFFT_FAIL(SHAFFT_ERR_PLAN_NOT_INIT);

    if (direction == shafft::FFTDirection::BACKWARD) {
      // Backward subplan ordering starts from the redistributed end state.
      // If forward output policy restored layout to INITIAL, re-apply
      // redistributions before executing inverse subplans.
      const int targetState = slab_->numExecSteps();
      while (slab_->esIndex() < targetState) {
        int rc = slab_->forward();
        if (rc != 0)
          return rc;
      }
    }

    void*data = nullptr, *work = nullptr;
    for (int i = 0; i < nsp; ++i) {
      const int subindex = (direction == shafft::FFTDirection::FORWARD) ? i : (nsp - 1 - i);

      slab_->getBuffers(&data, &work);
      if (!data || !work)
        SHAFFT_FAIL(SHAFFT_ERR_NO_BUFFER);

      SHAFFT_BACKEND_CHECK(fftndExecute(subplans[subindex], data, work, direction));

      // fftndExecute swaps pointers internally so result is always in 'data'
      // Update slab's buffer pointers to match
      slab_->setBuffers(data, work);

      if ((subindex < nsp - 1) && (direction == shafft::FFTDirection::FORWARD)) {
        int rc = slab_->forward();
        if (rc != 0)
          return rc;
      }
      if ((subindex > 0) && (direction == shafft::FFTDirection::BACKWARD)) {
        int rc = slab_->backward();
        if (rc != 0)
          return rc;
      }
    }

    if (direction == shafft::FFTDirection::FORWARD) {
      if (outputLayoutPolicy() == shafft::TransformLayout::INITIAL) {
        // Restore initial layout by replaying reverse redistribution stages.
        while (slab_->esIndex() > 0) {
          int rc = slab_->backward();
          if (rc != 0)
            return rc;
        }
      }
    }

    norm_exponent_ += 1;
    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

int shafft::detail::FFTNDPlan::normalizeImpl() noexcept {
  try {
    if (!slab_)
      SHAFFT_FAIL(SHAFFT_ERR_PLAN_NOT_INIT);

    int subsize[slab_->ndim()];
    slab_->getSubsize(subsize);
    long double normFactor = 1.0 / std::pow(norm_denominator, norm_exponent_);

    void*data = nullptr, *work = nullptr;
    slab_->getBuffers(&data, &work);
    if (!data || !work)
      SHAFFT_FAIL(SHAFFT_ERR_NO_BUFFER);

    const size_t tensorSize = detail::prodN<int, size_t>(subsize, slab_->ndim());
    int normRc = 0;

#if SHAFFT_BACKEND_HIPFFT
    hipStream_t stream = nullptr;
    if (subplans && nsubplans > 0) {
      stream = subplans[0].stream;
    }
    switch (fft_type) {
    case shafft::FFTType::C2C:
      normRc = normalizeComplexFloat((float)normFactor, tensorSize, data, stream);
      break;
    case shafft::FFTType::Z2Z:
      normRc = normalizeComplexDouble((double)normFactor, tensorSize, data, stream);
      break;
    }
    // Sync stream to ensure normalize is complete before returning
    if (normRc == 0) {
      hipError_t syncErr = hipStreamSynchronize(stream);
      if (syncErr != hipSuccess) {
        SHAFFT_FAIL_WITH(SHAFFT_ERR_BACKEND, SHAFFT_ERRSRC_HIP, static_cast<int>(syncErr));
      }
    }
#else
    switch (fft_type) {
    case shafft::FFTType::C2C:
      normRc = normalizeComplexFloat((float)normFactor, tensorSize, data);
      break;
    case shafft::FFTType::Z2Z:
      normRc = normalizeComplexDouble((double)normFactor, tensorSize, data);
      break;
    }
#endif

    if (normRc != 0) {
#if SHAFFT_BACKEND_HIPFFT
      SHAFFT_FAIL_WITH(SHAFFT_ERR_BACKEND, SHAFFT_ERRSRC_HIP, normRc);
#else
      SHAFFT_FAIL_WITH(SHAFFT_ERR_BACKEND, SHAFFT_ERRSRC_FFTW, normRc);
#endif
    }
    norm_exponent_ = 0;
    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

int shafft::detail::FFTNDPlan::createPlans() noexcept {
  // Delegate to the internal function
  return shafft::detail::planNDCreatePlans(this);
}

namespace shafft::detail {

#if SHAFFT_BACKEND_HIPFFT
int setStream(shafft::detail::FFTNDPlan* plan, hipStream_t stream) noexcept {
  try {
    if (!plan)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);
    if (!plan->subplans)
      SHAFFT_FAIL(SHAFFT_ERR_PLAN_NOT_INIT);
    for (int i = 0; i < plan->nsubplans; ++i) {
      SHAFFT_BACKEND_CHECK(fftndSetStream(plan->subplans[i], stream));
    }
    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}
#endif

int setBuffers(shafft::detail::FFTNDPlan* plan, void* data, void* work) noexcept;

int getBuffers(shafft::detail::FFTNDPlan* plan, void** data, void** work) noexcept;

int planCreate(shafft::detail::FFTNDPlan** plan) noexcept {
  if (!plan)
    SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);
  *plan = new (std::nothrow) shafft::detail::FFTNDPlan;
  if (!*plan) {
    SHAFFT_FAIL_WITH(SHAFFT_ERR_ALLOC, SHAFFT_ERRSRC_SYSTEM, 0);
  }
  return SHAFFT_STATUS(SHAFFT_SUCCESS);
}

// Plan *planCreate() { return new (std::nothrow) shafft::detail::FFTNDPlan; }

int planNDConfigure(shafft::detail::FFTNDPlan* plan,
                    int ndim,
                    int commDims[],
                    int dimensions[],
                    shafft::FFTType precision,
                    MPI_Comm comm) noexcept {
  try {
    if (!plan)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);

    // Inactive rank: caller passed MPI_COMM_NULL (e.g. from their own split).
    // Leave slab_ null, activeComm_ stays MPI_COMM_NULL.
    if (comm == MPI_COMM_NULL) {
      plan->comm = MPI_COMM_NULL;
      plan->nsubplans = 0;
      plan->subplans = nullptr;
      return SHAFFT_STATUS(SHAFFT_SUCCESS);
    }

    if (!commDims || !dimensions)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);
    if (ndim <= 0)
      SHAFFT_FAIL(SHAFFT_ERR_INVALID_DIM);
    if (commDims[ndim - 1] != 1)
      SHAFFT_FAIL(SHAFFT_ERR_INVALID_DECOMP);
    for (int i = 0; i < ndim; ++i)
      if (commDims[i] < 1 || commDims[i] > dimensions[i])
        SHAFFT_FAIL(SHAFFT_ERR_INVALID_DECOMP);

    // Count distributed axes (nda) for validation
    int ndaCount = 0;
    for (int i = 0; i < ndim; ++i) {
      if (commDims[i] > 1)
        ndaCount++;
      else
        break;
    }
    int ncaCount = ndim - ndaCount;

    // Require at least one contiguous (non-distributed) axis for slab decomposition
    if (ncaCount == 0)
      SHAFFT_FAIL(SHAFFT_ERR_INVALID_DECOMP);

    // Cross-axis constraint: during exchanges, commDims[j] partitions the contiguous
    // axis at position (ndim - nda + j). This must not exceed that axis's dimension.
    // commDims[j] <= dimensions[j + nca] for all j in [0, nda)
    for (int j = 0; j < ndaCount; ++j) {
      if (commDims[j] > dimensions[j + ncaCount])
        SHAFFT_FAIL(SHAFFT_ERR_INVALID_DECOMP);
    }

    if (precision != shafft::FFTType::C2C && precision != shafft::FFTType::Z2Z)
      SHAFFT_FAIL(SHAFFT_ERR_INVALID_FFTTYPE);

    // 1D tensors cannot be distributed: require single MPI rank
    int commSize = 1;
    MPI_Comm_size(comm, &commSize);
    if (ndim == 1 && commSize > 1)
      SHAFFT_FAIL(SHAFFT_ERR_INVALID_DECOMP); // Cannot distribute a 1D tensor

    // Element size for host-staging fallback (only used when SHAFFT_GPU_AWARE_MPI=0)
    size_t elemSize = (precision == shafft::FFTType::C2C) ? 2 * sizeof(float) : 2 * sizeof(double);

    switch (precision) {
    case shafft::FFTType::C2C:
      plan->fft_type = shafft::FFTType::C2C;
      plan->slab_ =
          new (std::nothrow) SlabND(ndim, dimensions, commDims, MPI_C_COMPLEX, comm, elemSize);
      break;
    case shafft::FFTType::Z2Z:
      plan->fft_type = shafft::FFTType::Z2Z;
      plan->slab_ = new (std::nothrow)
          SlabND(ndim, dimensions, commDims, MPI_C_DOUBLE_COMPLEX, comm, elemSize);
      break;
    default:
      SHAFFT_FAIL(SHAFFT_ERR_INVALID_FFTTYPE);
    }
    if (!plan->slab_) {
      SHAFFT_FAIL_WITH(SHAFFT_ERR_ALLOC, SHAFFT_ERRSRC_SYSTEM, 0);
    }

    // Duplicate the communicator so the plan owns its copy.
    SHAFFT_MPI_OR_FAIL(MPI_Comm_dup(comm, &plan->comm));

    // Handle inactive ranks: no FFT subplans needed, just warn and return success
    if (!plan->slab_->isActive()) {
      int worldRank = 0;
      MPI_Comm_rank(comm, &worldRank);
      std::fprintf(
          stderr,
          "[SHAFFT] Warning: rank %d is inactive and will not participate in FFT computation.\n",
          worldRank);
      plan->nsubplans = 0;
      plan->subplans = nullptr;
    }

    // Create the active subcommunicator (collective on plan->comm).
    // Inactive ranks (allocSize == 0) receive MPI_COMM_NULL.
    SHAFFT_MPI_OR_FAIL(plan->initActiveComm(plan->comm));

    plan->norm_denominator *= std::sqrt(detail::prodN<int, long double>(dimensions, ndim));
    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

int planNDCreatePlans(shafft::detail::FFTNDPlan* plan) noexcept {
  try {
    if (!plan)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);
    if (plan->planned_)
      SHAFFT_FAIL(SHAFFT_ERR_INVALID_STATE); // Already planned

    // Inactive rank: null-comm path (no slab created) or zero-alloc rank.
    if (!plan->slab_ || !plan->slab_->isActive()) {
      plan->planned_ = true;
      return SHAFFT_STATUS(SHAFFT_SUCCESS);
    }

    const int ndim = plan->slab_->ndim();
    plan->nsubplans = plan->slab_->nes() + 1;
    plan->subplans =
        new (std::nothrow) FFTNDHandle[plan->nsubplans](); // value-init to null members
    if (!plan->subplans) {
      SHAFFT_FAIL_WITH(SHAFFT_ERR_ALLOC, SHAFFT_ERRSRC_SYSTEM, 0);
    }

    int subsize[ndim], offset[ndim];
    const int nca = plan->slab_->nca();
    int ca[nca], ta[ndim], nta;
    int coordSpaces[ndim];
    for (int i = 0; i < ndim; ++i)
      coordSpaces[i] = 0;

    for (int i = 0; i < plan->nsubplans; ++i) {
      int cfgRc = plan->slab_->getIthConfig(subsize, offset, ca, i);
      if (cfgRc != 0)
        return cfgRc;
      getTransformAxes(coordSpaces, nca, ca, &nta, ta);
      plan->subplans[i].comm = plan->comm;

      void* data = nullptr;
      void* work = nullptr;
      plan->slab_->getBuffers(&data, &work);
      SHAFFT_BACKEND_CHECK(
          fftndPlan(plan->subplans[i], nta, ta, ndim, subsize, plan->fft_type, data, work));
    }

    plan->planned_ = true;
    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

// Note: ND-specific destroy, getLayout, getAxes, getAllocSize, execute, normalize
// functions have been removed in favor of the unified PlanBase* functions below.

// Unified functions (work on both ND and 1D plans via PlanBase*)

int destroy(shafft::detail::PlanBase** plan) noexcept {
  try {
    if (!plan || !*plan)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);

    (*plan)->release();
    delete *plan;
    *plan = nullptr;

    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

int execute(shafft::detail::PlanBase* plan, shafft::FFTDirection direction) noexcept {
  try {
    if (!plan)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);
    return plan->execute(direction);
  }
  SHAFFT_CATCH_RETURN();
}

int normalize(shafft::detail::PlanBase* plan) noexcept {
  try {
    if (!plan)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);
    if (!plan->isPlanned())
      SHAFFT_FAIL(SHAFFT_ERR_PLAN_NOT_INIT);
    return plan->normalize();
  }
  SHAFFT_CATCH_RETURN();
}

int getAllocSize(shafft::detail::PlanBase* plan, size_t* localAllocSize) noexcept {
  try {
    if (!localAllocSize)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);
    if (!plan || !plan->isConfigured())
      SHAFFT_FAIL(SHAFFT_ERR_PLAN_NOT_INIT);
    *localAllocSize = plan->allocSize();
    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

int getGlobalSize(shafft::detail::PlanBase* plan, size_t* globalSize) noexcept {
  try {
    if (!globalSize)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);
    if (!plan || !plan->isConfigured())
      SHAFFT_FAIL(SHAFFT_ERR_PLAN_NOT_INIT);
    *globalSize = plan->globalSize();
    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

int getLayout(shafft::detail::PlanBase* plan,
              size_t* subsize,
              size_t* offset,
              shafft::TensorLayout layout) noexcept {
  try {
    if (!plan || !plan->isConfigured())
      SHAFFT_FAIL(SHAFFT_ERR_PLAN_NOT_INIT);
    if (!subsize || !offset)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);
    return plan->getLayout(subsize, offset, layout);
  }
  SHAFFT_CATCH_RETURN();
}

int getAxes(shafft::detail::PlanBase* plan,
            int* ca,
            int* da,
            shafft::TensorLayout layout) noexcept {
  try {
    if (!plan || !plan->isConfigured())
      SHAFFT_FAIL(SHAFFT_ERR_PLAN_NOT_INIT);
    // Note: ca or da may be null if nca() or nda() is 0 (e.g., FFT1D has nca=0)
    return plan->getAxes(ca, da, layout);
  }
  SHAFFT_CATCH_RETURN();
}

int isConfigured(shafft::detail::PlanBase* plan, int* configured) noexcept {
  try {
    if (!configured)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);
    *configured = (plan && plan->isConfigured()) ? 1 : 0;
    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

int isActive(shafft::detail::PlanBase* plan, int* isActive) noexcept {
  try {
    if (!isActive)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);
    *isActive = (plan && plan->isConfigured() && plan->isActive()) ? 1 : 0;
    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

int getPrecision(shafft::detail::PlanBase* plan, shafft::FFTType* precision) noexcept {
  try {
    if (!precision)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);
    if (!plan || !plan->isConfigured())
      SHAFFT_FAIL(SHAFFT_ERR_PLAN_NOT_INIT);
    *precision = plan->precision();
    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

int setBuffers(shafft::detail::PlanBase* plan, void* data, void* work) noexcept {
  try {
    if (!plan)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);
    if (!plan->isConfigured())
      SHAFFT_FAIL(SHAFFT_ERR_PLAN_NOT_INIT);
    const bool inactive = !plan->isActive() || plan->allocSize() == 0;
    if (inactive) {
      // Inactive ranks are no-ops; allow null buffers while keeping active ranks strict.
      plan->setBuffers(nullptr, nullptr);
      return SHAFFT_STATUS(SHAFFT_SUCCESS);
    }
    if (!data || !work)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);
    plan->setBuffers(data, work);
    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

int getBuffers(shafft::detail::PlanBase* plan, void** data, void** work) noexcept {
  try {
    if (!plan)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);
    if (!plan->isConfigured())
      SHAFFT_FAIL(SHAFFT_ERR_PLAN_NOT_INIT);
    if (!data || !work)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);

    const bool inactive = !plan->isActive() || plan->allocSize() == 0;
    plan->getBuffers(data, work);

    // Inactive ranks may have null buffers (consistent with setBuffers behavior)
    if (!inactive && (!*data || !*work))
      SHAFFT_FAIL(SHAFFT_ERR_NO_BUFFER);
    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

#if SHAFFT_BACKEND_HIPFFT
int setStream(shafft::detail::PlanBase* plan, hipStream_t stream) noexcept {
  try {
    if (!plan)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);
    if (!plan->isConfigured())
      SHAFFT_FAIL(SHAFFT_ERR_PLAN_NOT_INIT);
    plan->setStream(static_cast<void*>(stream));
    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}
#endif

} // namespace shafft::detail

// FFT1D (1-dimensional distributed FFT) implementation

// FFT1DPlan method implementations (struct defined in <shafft/detail/plan_1d_data.hpp>)
void shafft::detail::FFT1DPlan::release() noexcept {
  if (initialized) {
    if (handle) {
      fft1dDestroy(*handle);
      delete handle;
      handle = nullptr;
    }
    initialized = false;
  }
  freeActiveComm();

  // Free the plan's owned communicator (dup'd during configure).
  if (comm_ != MPI_COMM_NULL) {
    MPI_Comm_free(&comm_);
    comm_ = MPI_COMM_NULL;
  }

  norm_exponent_ = 0;
}

int shafft::detail::FFT1DPlan::executeImpl(shafft::FFTDirection direction) noexcept {
  try {
    if (!initialized)
      SHAFFT_FAIL(SHAFFT_ERR_PLAN_NOT_INIT);
    if (!handle)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);

    // Get buffers from slab
    void* dataBuf = nullptr;
    void* workBuf = nullptr;
    slab1d_.getBuffers(&dataBuf, &workBuf);
    if (!dataBuf)
      SHAFFT_FAIL(SHAFFT_ERR_NO_BUFFER);

    // Use work buffer if set, otherwise in-place
    void* work = workBuf ? workBuf : dataBuf;
    // Execute: reads from dataBuf, writes to work
    int rc = fft1dExecute(*handle, dataBuf, work, direction);
    if (rc == 0) {
      // Swap buffers so data_ points to the result (consistent with FFTND behavior)
      // After this, getBuffers() returns (result, input) instead of (input, result)
      slab1d_.swapBuffers();

      // Update slab state (0 = INITIAL, 1 = REDISTRIBUTED)
      slab1d_.setState(direction == shafft::FFTDirection::FORWARD ? 1 : 0);
      // Track normalization exponent
      norm_exponent_ += 1;
    }
    return rc;
  }
  SHAFFT_CATCH_RETURN();
}

int shafft::detail::FFT1DPlan::normalizeImpl() noexcept {
  try {
    if (!initialized)
      SHAFFT_FAIL(SHAFFT_ERR_PLAN_NOT_INIT);
    if (!handle)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);

    // Get buffers from slab - after execute(), data_ contains the result
    void* dataBuf = nullptr;
    void* workBuf = nullptr;
    slab1d_.getBuffers(&dataBuf, &workBuf);
    if (!dataBuf)
      SHAFFT_FAIL(SHAFFT_ERR_NO_BUFFER);

    // Normalize the data buffer (where result is after buffer swap)
    // Use 1/pow(sqrt(N), norm_exponent) like FFTND
    int rc = fft1dNormalize(*handle, dataBuf, norm_exponent_);
    norm_exponent_ = 0; // Reset after normalization
    return rc;
  }
  SHAFFT_CATCH_RETURN();
}

int shafft::detail::FFT1DPlan::createPlans() noexcept {
  // Delegate to the internal function
  return shafft::detail::fft1dCreatePlans(this);
}

namespace shafft::detail {

int configuration1D(size_t globalN,
                    size_t* localN,
                    size_t* localStart,
                    size_t* localAllocSize,
                    shafft::FFTType precision,
                    MPI_Comm comm) noexcept {
  try {
    if (!localN || !localStart || !localAllocSize)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);
    (void)precision; // Backend computes uniform distribution regardless of precision

    size_t ln = 0, ls = 0, as = 0;
    int rc = fft1dQueryLayout(globalN, ln, ls, as, comm);
    if (rc != 0)
      return rc;

    *localN = ln;
    *localStart = ls;
    *localAllocSize = as;
    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

int fft1dCreate(shafft::detail::FFT1DPlan** plan) noexcept {
  try {
    if (!plan)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);
    *plan = new (std::nothrow) shafft::detail::FFT1DPlan();
    if (!*plan)
      SHAFFT_FAIL_WITH(SHAFFT_ERR_ALLOC, SHAFFT_ERRSRC_SYSTEM, 0);
    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

int fft1dConfigure(shafft::detail::FFT1DPlan* plan,
                   size_t globalN,
                   size_t localN,
                   size_t localStart,
                   shafft::FFTType precision,
                   MPI_Comm comm) noexcept {
  try {
    if (!plan)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);
    if (plan->configured)
      SHAFFT_FAIL(SHAFFT_ERR_INVALID_STATE); // Already configured

    // Inactive rank: caller passed MPI_COMM_NULL (e.g. from their own split).
    if (comm == MPI_COMM_NULL) {
      plan->comm_ = MPI_COMM_NULL;
      plan->configured = true;
      return SHAFFT_STATUS(SHAFFT_SUCCESS);
    }

    if (precision != shafft::FFTType::C2C && precision != shafft::FFTType::Z2Z)
      SHAFFT_FAIL(SHAFFT_ERR_INVALID_FFTTYPE);

    // localN and localStart are passed for validation but computed by backend
    (void)localN;
    (void)localStart;

    // Query layout from backend without creating plans
    size_t queryLocalN, queryLocalStart, queryAllocSize;
    int rc = fft1dQueryLayout(globalN, queryLocalN, queryLocalStart, queryAllocSize, comm);
    if (rc != 0)
      return rc;

    // Initialize Slab1D with queried layout
    // Note: For 1D FFT, initial and redistributed layouts are the same
    plan->slab1d_.init(globalN,
                       queryLocalN,
                       queryLocalStart,
                       queryLocalN,     // local_n_trans same as local_n_init for 1D
                       queryLocalStart, // local_start_trans same as local_start_init for 1D
                       queryAllocSize,
                       precision);

    // Store parameters for later plan creation.
    // Duplicate the communicator so the plan owns its copy.
    plan->globalN_ = globalN;
    SHAFFT_MPI_OR_FAIL(MPI_Comm_dup(comm, &plan->comm_));
    plan->precision = precision;
    plan->configured = true;

    // Create the active subcommunicator (collective on plan->comm_).
    // Inactive ranks (allocSize == 0) receive MPI_COMM_NULL.
    SHAFFT_MPI_OR_FAIL(plan->initActiveComm(plan->comm_));

    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

int fft1dCreatePlans(shafft::detail::FFT1DPlan* plan) noexcept {
  try {
    if (!plan)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);
    if (!plan->configured)
      SHAFFT_FAIL(SHAFFT_ERR_PLAN_NOT_INIT); // Not configured
    if (plan->initialized)
      SHAFFT_FAIL(SHAFFT_ERR_INVALID_STATE); // Already planned

    // Inactive rank (null-comm path): nothing to plan.
    if (plan->comm_ == MPI_COMM_NULL) {
      plan->initialized = true;
      return SHAFFT_STATUS(SHAFFT_SUCCESS);
    }

    plan->handle = new FFT1DHandle{};

    void* data = nullptr;
    void* work = nullptr;
    plan->slab1d_.getBuffers(&data, &work);
    int rc = fft1dPlan(*plan->handle, plan->globalN_, plan->precision, plan->comm_, data, work);
    if (rc != 0) {
      delete plan->handle;
      plan->handle = nullptr;
      return rc;
    }

    plan->initialized = true;
    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

} // namespace shafft::detail
