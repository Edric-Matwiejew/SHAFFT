// C interface implementation (updated)
#include "../cpp/detail/config_info.hpp"
#include "../cpp/detail/error_macros.hpp"
#include "../cpp/detail/plan_data_base.hpp" // for unified PlanBase interface
#include <shafft/shafft.h>
#include <shafft/shafft.hpp> // for portable buffer functions

#include "shafft_internal.hpp"

// ---- Safe casters (no exceptions) ------------------------------------------

static int shafftCTypeCast(shafft_t cType, shafft::FFTType* out) {
  if (!out)
    return (int)shafft::Status::ERR_NULLPTR;
  switch (cType) {
  case SHAFFT_C2C:
    *out = shafft::FFTType::C2C;
    return (int)shafft::Status::SUCCESS;
  case SHAFFT_Z2Z:
    *out = shafft::FFTType::Z2Z;
    return (int)shafft::Status::SUCCESS;
  default:
    return (int)shafft::Status::ERR_INVALID_FFTTYPE;
  }
}

static int shafftDirectionCast(shafft_direction_t cdir, shafft::FFTDirection* out) {
  if (!out)
    return (int)shafft::Status::ERR_NULLPTR;
  switch (cdir) {
  case SHAFFT_FORWARD:
    *out = shafft::FFTDirection::FORWARD;
    return (int)shafft::Status::SUCCESS;
  case SHAFFT_BACKWARD:
    *out = shafft::FFTDirection::BACKWARD;
    return (int)shafft::Status::SUCCESS;
  default:
    return (int)shafft::Status::ERR_INVALID_DIM;
  }
}

static int shafftTransformLayoutCast(shafft_transform_layout_t policy,
                                     shafft::TransformLayout* out) {
  if (!out)
    return (int)shafft::Status::ERR_NULLPTR;
  switch (policy) {
  case SHAFFT_LAYOUT_REDISTRIBUTED:
    *out = shafft::TransformLayout::REDISTRIBUTED;
    return (int)shafft::Status::SUCCESS;
  case SHAFFT_LAYOUT_INITIAL:
    *out = shafft::TransformLayout::INITIAL;
    return (int)shafft::Status::SUCCESS;
  default:
    return (int)shafft::Status::ERR_INVALID_LAYOUT;
  }
}

// Plan-type specific functions (ND)

int shafftNDCreate(void** outPlan) {
  try {
    if (!outPlan)
      return (int)shafft::Status::ERR_NULLPTR;
    *outPlan = nullptr;
    shafft::detail::FFTNDPlan* p = nullptr;
    int rc = shafft::detail::planCreate(&p);
    if (rc == (int)shafft::Status::SUCCESS)
      *outPlan = static_cast<void*>(p);
    return rc;
  }
  SHAFFT_CATCH_RETURN();
}

int shafftNDInit(void* planPtr,
                 int ndim,
                 int commDims[],
                 int dimensions[],
                 shafft_t precision,
                 MPI_Comm cComm,
                 shafft_transform_layout_t output_policy) {
  try {
    if (!planPtr)
      return (int)shafft::Status::ERR_NULLPTR;
    shafft::detail::FFTNDPlan* plan = static_cast<shafft::detail::FFTNDPlan*>(planPtr);

    shafft::FFTType prec;
    int rc = shafftCTypeCast(precision, &prec);
    if (rc != (int)shafft::Status::SUCCESS)
      return rc;

    shafft::TransformLayout policy;
    rc = shafftTransformLayoutCast(output_policy, &policy);
    if (rc != (int)shafft::Status::SUCCESS)
      return rc;

    plan->setOutputLayoutPolicy(policy);

    // Configure only - call shafftPlan() afterward to create backend plans
    return shafft::detail::planNDConfigure(plan, ndim, commDims, dimensions, prec, cComm);
  }
  SHAFFT_CATCH_RETURN();
}

int shafftConfigurationND(int ndim,
                          int* size,
                          shafft_t precision,
                          int* commDims,
                          int* nda,
                          size_t* subsize,
                          size_t* offset,
                          int* commSize,
                          shafft_decomposition_strategy_t strategy,
                          size_t memLimit,
                          MPI_Comm cComm) {
  try {
    shafft::FFTType prec;
    int rc = shafftCTypeCast(precision, &prec);
    if (rc != (int)shafft::Status::SUCCESS)
      return rc;

    // Map C enum to C++ enum
    shafft::DecompositionStrategy strat;
    switch (strategy) {
    case SHAFFT_MAXIMIZE_NDA:
      strat = shafft::DecompositionStrategy::MAXIMIZE_NDA;
      break;
    case SHAFFT_MINIMIZE_NDA:
      strat = shafft::DecompositionStrategy::MINIMIZE_NDA;
      break;
    default:
      strat = shafft::DecompositionStrategy::MINIMIZE_NDA;
      break;
    }

    // Internal API uses int, convert on output
    std::vector<int> intSubsize(ndim), intOffset(ndim);
    rc = shafft::detail::configurationND(ndim,
                                         size,
                                         prec,
                                         commDims,
                                         nda,
                                         intSubsize.data(),
                                         intOffset.data(),
                                         commSize,
                                         strat,
                                         memLimit,
                                         cComm);
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

int shafftPlan(void* planPtr) {
  try {
    if (!planPtr)
      return (int)shafft::Status::ERR_NULLPTR;
    shafft::detail::PlanBase* base = static_cast<shafft::detail::PlanBase*>(planPtr);
    return base->createPlans();
  }
  SHAFFT_CATCH_RETURN();
}

// Plan-type specific functions (1D)

int shafftConfiguration1D(size_t globalN,
                          size_t* localN,
                          size_t* localStart,
                          size_t* localAllocSize,
                          shafft_t precision,
                          MPI_Comm cComm) {
  try {
    shafft::FFTType prec;
    int rc = shafftCTypeCast(precision, &prec);
    if (rc != (int)shafft::Status::SUCCESS)
      return rc;
    return shafft::detail::configuration1D(
        globalN, localN, localStart, localAllocSize, prec, cComm);
  }
  SHAFFT_CATCH_RETURN();
}

int shafft1DCreate(void** outPlan) {
  try {
    if (!outPlan)
      return (int)shafft::Status::ERR_NULLPTR;
    *outPlan = nullptr;
    shafft::detail::FFT1DPlan* p = nullptr;
    int rc = shafft::detail::fft1dCreate(&p);
    if (rc == (int)shafft::Status::SUCCESS)
      *outPlan = static_cast<void*>(p);
    return rc;
  }
  SHAFFT_CATCH_RETURN();
}

int shafft1DInit(void* planPtr,
                 size_t globalN,
                 size_t localN,
                 size_t localStart,
                 shafft_t precision,
                 MPI_Comm cComm) {
  try {
    if (!planPtr)
      return (int)shafft::Status::ERR_NULLPTR;
    shafft::detail::FFT1DPlan* plan = static_cast<shafft::detail::FFT1DPlan*>(planPtr);

    shafft::FFTType prec;
    int rc = shafftCTypeCast(precision, &prec);
    if (rc != (int)shafft::Status::SUCCESS)
      return rc;

    // Configure only - call shafftPlan() afterward to create backend plans
    return shafft::detail::fft1dConfigure(plan, globalN, localN, localStart, prec, cComm);
  }
  SHAFFT_CATCH_RETURN();
}

// Unified functions (work on both ND and 1D plans via PlanBase*)

int shafftDestroy(void** planPtr) {
  try {
    if (!planPtr)
      return (int)shafft::Status::ERR_NULLPTR;
    auto* base = static_cast<shafft::detail::PlanBase*>(*planPtr);
    int rc = shafft::detail::destroy(&base);
    if (rc == static_cast<int>(shafft::Status::SUCCESS)) {
      *planPtr = nullptr;
    }
    return rc;
  }
  SHAFFT_CATCH_RETURN();
}

int shafftSetBuffers(void* planPtr, void* data, void* work) {
  try {
    if (!planPtr)
      return static_cast<int>(shafft::Status::ERR_NULLPTR);
    auto* base = static_cast<shafft::detail::PlanBase*>(planPtr);
    return shafft::detail::setBuffers(base, data, work);
  }
  SHAFFT_CATCH_RETURN();
}

int shafftGetBuffers(void* planPtr, void** data, void** work) {
  try {
    if (!planPtr || !data || !work)
      return static_cast<int>(shafft::Status::ERR_NULLPTR);
    auto* base = static_cast<shafft::detail::PlanBase*>(planPtr);
    return shafft::detail::getBuffers(base, data, work);
  }
  SHAFFT_CATCH_RETURN();
}

int shafftGetLayout(void* planPtr, size_t* subsize, size_t* offset, shafft_tensor_layout_t layout) {
  try {
    if (!planPtr)
      return (int)shafft::Status::ERR_NULLPTR;
    auto* base = static_cast<shafft::detail::PlanBase*>(planPtr);
    return shafft::detail::getLayout(
        base, subsize, offset, static_cast<shafft::TensorLayout>(layout));
  }
  SHAFFT_CATCH_RETURN();
}

int shafftGetAxes(void* planPtr, int* ca, int* da, shafft_tensor_layout_t layout) {
  try {
    if (!planPtr)
      return (int)shafft::Status::ERR_NULLPTR;
    auto* base = static_cast<shafft::detail::PlanBase*>(planPtr);
    return shafft::detail::getAxes(base, ca, da, static_cast<shafft::TensorLayout>(layout));
  }
  SHAFFT_CATCH_RETURN();
}

int shafftGetAllocSize(void* planPtr, size_t* localAllocSize) {
  try {
    if (!planPtr || !localAllocSize)
      return (int)shafft::Status::ERR_NULLPTR;
    auto* base = static_cast<shafft::detail::PlanBase*>(planPtr);
    return shafft::detail::getAllocSize(base, localAllocSize);
  }
  SHAFFT_CATCH_RETURN();
}

int shafftGetGlobalSize(void* planPtr, size_t* globalSize) {
  try {
    if (!planPtr || !globalSize)
      return (int)shafft::Status::ERR_NULLPTR;
    auto* base = static_cast<shafft::detail::PlanBase*>(planPtr);
    return shafft::detail::getGlobalSize(base, globalSize);
  }
  SHAFFT_CATCH_RETURN();
}

int shafftIsConfigured(void* planPtr, int* configured) {
  try {
    auto* base = static_cast<shafft::detail::PlanBase*>(planPtr);
    return shafft::detail::isConfigured(base, configured);
  }
  SHAFFT_CATCH_RETURN();
}

int shafftIsActive(void* planPtr, int* active) {
  try {
    auto* base = static_cast<shafft::detail::PlanBase*>(planPtr);
    return shafft::detail::isActive(base, active);
  }
  SHAFFT_CATCH_RETURN();
}

int shafftExecute(void* planPtr, shafft_direction_t direction) {
  try {
    if (!planPtr)
      return (int)shafft::Status::ERR_NULLPTR;

    shafft::FFTDirection dir;
    int rc = shafftDirectionCast(direction, &dir);
    if (rc != (int)shafft::Status::SUCCESS)
      return rc;

    auto* base = static_cast<shafft::detail::PlanBase*>(planPtr);
    return shafft::detail::execute(base, dir);
  }
  SHAFFT_CATCH_RETURN();
}

int shafftNormalize(void* planPtr) {
  try {
    if (!planPtr)
      return (int)shafft::Status::ERR_NULLPTR;
    auto* base = static_cast<shafft::detail::PlanBase*>(planPtr);
    return shafft::detail::normalize(base);
  }
  SHAFFT_CATCH_RETURN();
}

#if SHAFFT_BACKEND_HIPFFT
int shafftSetStream(void* planPtr, hipStream_t stream) {
  try {
    if (!planPtr)
      return (int)shafft::Status::ERR_NULLPTR;
    auto* base = static_cast<shafft::detail::PlanBase*>(planPtr);
    return shafft::detail::setStream(base, stream);
  }
  SHAFFT_CATCH_RETURN();
}
#endif

// Portable buffer allocation/free

int shafftAllocBufferF(size_t count, void** buf) {
  try {
    if (!buf)
      return (int)shafft::Status::ERR_NULLPTR;
    shafft::complexf* ptr = nullptr;
    int rc = shafft::allocBuffer(count, &ptr);
    *buf = static_cast<void*>(ptr);
    return rc;
  }
  SHAFFT_CATCH_RETURN();
}

int shafftAllocBufferD(size_t count, void** buf) {
  try {
    if (!buf)
      return (int)shafft::Status::ERR_NULLPTR;
    shafft::complexd* ptr = nullptr;
    int rc = shafft::allocBuffer(count, &ptr);
    *buf = static_cast<void*>(ptr);
    return rc;
  }
  SHAFFT_CATCH_RETURN();
}

int shafftFreeBufferF(void* buf) {
  try {
    return shafft::freeBuffer(static_cast<shafft::complexf*>(buf));
  }
  SHAFFT_CATCH_RETURN();
}

int shafftFreeBufferD(void* buf) {
  try {
    return shafft::freeBuffer(static_cast<shafft::complexd*>(buf));
  }
  SHAFFT_CATCH_RETURN();
}

// Portable memory copy

int shafftCopyToBufferF(void* dst, const void* src, size_t count) {
  try {
    return shafft::copyToBuffer(
        static_cast<shafft::complexf*>(dst), static_cast<const shafft::complexf*>(src), count);
  }
  SHAFFT_CATCH_RETURN();
}

int shafftCopyToBufferD(void* dst, const void* src, size_t count) {
  try {
    return shafft::copyToBuffer(
        static_cast<shafft::complexd*>(dst), static_cast<const shafft::complexd*>(src), count);
  }
  SHAFFT_CATCH_RETURN();
}

int shafftCopyFromBufferF(void* dst, const void* src, size_t count) {
  try {
    return shafft::copyFromBuffer(
        static_cast<shafft::complexf*>(dst), static_cast<const shafft::complexf*>(src), count);
  }
  SHAFFT_CATCH_RETURN();
}

int shafftCopyFromBufferD(void* dst, const void* src, size_t count) {
  try {
    return shafft::copyFromBuffer(
        static_cast<shafft::complexd*>(dst), static_cast<const shafft::complexd*>(src), count);
  }
  SHAFFT_CATCH_RETURN();
}

// ---- Config object lifecycle -----------------------------------------------
// Note: shafftConfigNDInit, shafftConfigNDRelease, shafftConfigNDResolve,
//       shafftConfig1DInit, shafftConfig1DRelease, shafftConfig1DResolve
// are defined in config_info.cpp (linked into the shafft OBJECT lib).

int shafftNDInitFromConfig(void* plan, shafft_nd_config_t* cfg) {
  try {
    if (!plan || !cfg)
      return (int)shafft::Status::ERR_NULLPTR;
    if (cfg->structSize != sizeof(shafft_nd_config_t))
      return (int)shafft::Status::ERR_INVALID_STATE;

    // Auto-resolve if not yet resolved
    if (!(cfg->flags & SHAFFT_CONFIG_RESOLVED)) {
      int rc = shafft::detail::configNDResolve(cfg);
      if (rc != 0)
        return rc;
    }

    // Use the active subcommunicator created by resolve.
    // Inactive ranks have MPI_COMM_NULL, which planNDConfigure handles.
    MPI_Comm comm = cfg->activeComm;

    // Delegate to the existing init path using resolved values from the config
    auto* p = static_cast<shafft::detail::PlanBase*>(plan);
    auto* ndp = dynamic_cast<shafft::detail::FFTNDPlan*>(p);
    if (!ndp)
      return (int)shafft::Status::ERR_INVALID_STATE;

    // Convert size_t globalShape to int array
    std::vector<int> intShape(cfg->ndim);
    for (int i = 0; i < cfg->ndim; ++i)
      intShape[i] = static_cast<int>(cfg->globalShape[i]);

    // Use resolved commDims from config
    std::vector<int> commDims(cfg->commDims, cfg->commDims + cfg->ndim);

    shafft::FFTType prec =
        (cfg->precision == SHAFFT_Z2Z) ? shafft::FFTType::Z2Z : shafft::FFTType::C2C;

    // Apply output layout policy from the config struct.
    shafft::TransformLayout policy = (cfg->outputPolicy == SHAFFT_LAYOUT_INITIAL)
                                         ? shafft::TransformLayout::INITIAL
                                         : shafft::TransformLayout::REDISTRIBUTED;
    ndp->setOutputLayoutPolicy(policy);

    return shafft::detail::planNDConfigure(
        ndp, cfg->ndim, commDims.data(), intShape.data(), prec, comm);
  }
  SHAFFT_CATCH_RETURN();
}

int shafft1DInitFromConfig(void* plan, shafft_1d_config_t* cfg) {
  try {
    if (!plan || !cfg)
      return (int)shafft::Status::ERR_NULLPTR;
    if (cfg->structSize != sizeof(shafft_1d_config_t))
      return (int)shafft::Status::ERR_INVALID_STATE;

    // Auto-resolve if not yet resolved
    if (!(cfg->flags & SHAFFT_CONFIG_RESOLVED)) {
      int rc = shafft::detail::config1DResolve(cfg);
      if (rc != 0)
        return rc;
    }

    // Use the active subcommunicator created by resolve.
    // Inactive ranks have MPI_COMM_NULL, which fft1dConfigure handles.
    MPI_Comm comm = cfg->activeComm;

    auto* p = static_cast<shafft::detail::PlanBase*>(plan);
    auto* p1d = dynamic_cast<shafft::detail::FFT1DPlan*>(p);
    if (!p1d)
      return (int)shafft::Status::ERR_INVALID_STATE;

    shafft::FFTType prec =
        (cfg->precision == SHAFFT_Z2Z) ? shafft::FFTType::Z2Z : shafft::FFTType::C2C;
    return shafft::detail::fft1dConfigure(
        p1d, cfg->globalSize, cfg->initial.localSize, cfg->initial.localStart, prec, comm);
  }
  SHAFFT_CATCH_RETURN();
}

int shafftGetCommunicator(void* plan, MPI_Comm* outComm) {
  try {
    if (!plan || !outComm)
      return (int)shafft::Status::ERR_NULLPTR;

    auto* p = static_cast<shafft::detail::PlanBase*>(plan);
    MPI_Comm src = p->getComm();
    if (src == MPI_COMM_NULL) {
      *outComm = MPI_COMM_NULL;
    } else {
      int rc = MPI_Comm_dup(src, outComm);
      if (rc != MPI_SUCCESS)
        return (int)shafft::Status::ERR_MPI;
    }
    return (int)shafft::Status::SUCCESS;
  }
  SHAFFT_CATCH_RETURN();
}

// Library information

const char* shafftGetBackendName(void) {
  return shafft::getBackendName();
}

void shafftGetVersion(int* major, int* minor, int* patch) {
  shafft::Version v = shafft::getVersion();
  if (major)
    *major = v.major;
  if (minor)
    *minor = v.minor;
  if (patch)
    *patch = v.patch;
}

const char* shafftGetVersionString(void) {
  return shafft::getVersionString();
}

// Finalization

int shafftFinalize(void) {
  return shafft::finalize();
}
