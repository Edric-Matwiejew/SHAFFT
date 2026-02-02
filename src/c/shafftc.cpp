// C interface implementation (updated)
#include <shafft/shafft.h>
#include <shafft/shafft.hpp>        // for portable buffer functions
#include <shafft/shafft_error.hpp>  // for SHAFFT_STATUS codes

#include "_shafft.hpp"

#if SHAFFT_BACKEND_HIPFFT
int shafftSetStream(void* plan_ptr, hipStream_t stream) {
  if (!plan_ptr)
    return (int)shafft::Status::SHAFFT_ERR_NULLPTR;
  shafft::PlanData* plan = static_cast<shafft::PlanData*>(plan_ptr);
  return _shafft::setStream(plan, stream);
}
#endif

// ---- Safe casters (no exceptions) ------------------------------------------

static int shafftCTypeCast(shafft_t cType, shafft::FFTType* out) {
  if (!out)
    return (int)shafft::Status::SHAFFT_ERR_NULLPTR;
  switch (cType) {
    case SHAFFT_C2C:
      *out = shafft::FFTType::C2C;
      return (int)shafft::Status::SHAFFT_SUCCESS;
    case SHAFFT_Z2Z:
      *out = shafft::FFTType::Z2Z;
      return (int)shafft::Status::SHAFFT_SUCCESS;
    default:
      return (int)shafft::Status::SHAFFT_ERR_INVALID_FFTTYPE;
  }
}

static int fftCDirectionCast(shafft_direction_t cdir, shafft::FFTDirection* out) {
  if (!out)
    return (int)shafft::Status::SHAFFT_ERR_NULLPTR;
  switch (cdir) {
    case SHAFFT_FORWARD:
      *out = shafft::FFTDirection::FORWARD;
      return (int)shafft::Status::SHAFFT_SUCCESS;
    case SHAFFT_BACKWARD:
      *out = shafft::FFTDirection::BACKWARD;
      return (int)shafft::Status::SHAFFT_SUCCESS;
    default:
      return (int)shafft::Status::SHAFFT_ERR_INVALID_DIM;  // or define a DIR error
  }
}

// ---- Lifecycle --------------------------------------------------------------

int shafftPlanCreate(void** out_plan) {
  if (!out_plan)
    return (int)shafft::Status::SHAFFT_ERR_NULLPTR;
  *out_plan = nullptr;
  shafft::PlanData* p = nullptr;
  int rc = _shafft::planCreate(&p);
  if (rc == (int)shafft::Status::SHAFFT_SUCCESS)
    *out_plan = static_cast<void*>(p);
  return rc;
}

int shafftDestroy(void** plan_ptr) {
  if (!plan_ptr)
    return (int)shafft::Status::SHAFFT_ERR_NULLPTR;
  shafft::PlanData* plan = static_cast<shafft::PlanData*>(*plan_ptr);
  int rc = _shafft::destroy(&plan);
  if (rc == static_cast<int>(shafft::Status::SHAFFT_SUCCESS)) {
    *plan_ptr = nullptr;
  }
  return rc;
}

int shafftSetBuffers(void* plan_ptr, void* data, void* work) {
  if (!plan_ptr || !data || !work)
    return static_cast<int>(shafft::Status::SHAFFT_ERR_NULLPTR);
  shafft::PlanData* plan = static_cast<shafft::PlanData*>(plan_ptr);
  int rc = _shafft::setBuffers(plan, data, work);
  return rc;
}

int shafftGetBuffers(void* plan_ptr, void** data, void** work) {
  if (!plan_ptr || !data || !work)
    return static_cast<int>(shafft::Status::SHAFFT_ERR_NULLPTR);
  shafft::PlanData* plan = static_cast<shafft::PlanData*>(plan_ptr);
  int rc = _shafft::getBuffers(plan, data, work);
  return rc;
}

// ---- Configuration ----------------------------------------------------------

int shafftConfigurationNDA(int ndim, int* size, int* nda, int* subsize, int* offset, int* COMM_DIMS,
                           shafft_t precision, size_t mem_limit, MPI_Comm c_comm) {
  shafft::FFTType prec;
  int rc = shafftCTypeCast(precision, &prec);
  if (rc != (int)shafft::Status::SHAFFT_SUCCESS)
    return rc;

  return _shafft::configurationNDA(ndim, size, nda, subsize, offset, COMM_DIMS, prec, mem_limit,
                                   c_comm);
}

int shafftConfigurationCart(int ndim, int* size, int* subsize, int* offset, int* COMM_DIMS,
                            int* COMM_SIZE, shafft_t precision, size_t mem_limit, MPI_Comm c_comm) {
  shafft::FFTType prec;
  int rc = shafftCTypeCast(precision, &prec);
  if (rc != (int)shafft::Status::SHAFFT_SUCCESS)
    return rc;

  return _shafft::configurationCart(ndim, size, subsize, offset, COMM_DIMS, COMM_SIZE, prec,
                                    mem_limit, c_comm);
}

// ---- Planning ---------------------------------------------------------------

int shafftPlanNDA(void* plan_ptr, int ndim, int nda, int dimensions[], shafft_t precision,
                  MPI_Comm c_comm) {
  if (!plan_ptr)
    return (int)shafft::Status::SHAFFT_ERR_NULLPTR;
  shafft::PlanData* plan = static_cast<shafft::PlanData*>(plan_ptr);

  shafft::FFTType prec;
  int rc = shafftCTypeCast(precision, &prec);
  if (rc != (int)shafft::Status::SHAFFT_SUCCESS)
    return rc;

  return _shafft::planNDA(plan, ndim, nda, dimensions, prec, c_comm);
}

int shafftPlanCart(void* plan_ptr, int ndim, int COMM_DIMS[], int dimensions[], shafft_t precision,
                   MPI_Comm c_comm) {
  if (!plan_ptr)
    return (int)shafft::Status::SHAFFT_ERR_NULLPTR;
  shafft::PlanData* plan = static_cast<shafft::PlanData*>(plan_ptr);

  shafft::FFTType prec;
  int rc = shafftCTypeCast(precision, &prec);
  if (rc != (int)shafft::Status::SHAFFT_SUCCESS)
    return rc;

  return _shafft::planCart(plan, ndim, COMM_DIMS, dimensions, prec, c_comm);
}

// ---- Queries/exec -----------------------------------------------------------

int shafftGetLayout(void* plan_ptr, int* subsize, int* offset, shafft_tensor_layout_t layout) {
  if (!plan_ptr)
    return (int)shafft::Status::SHAFFT_ERR_NULLPTR;
  shafft::PlanData* plan = static_cast<shafft::PlanData*>(plan_ptr);
  return _shafft::getLayout(plan, subsize, offset, static_cast<shafft::TensorLayout>(layout));
}

int shafftGetAxes(void* plan_ptr, int* ca, int* da, shafft_tensor_layout_t layout) {
  if (!plan_ptr)
    return (int)shafft::Status::SHAFFT_ERR_NULLPTR;
  shafft::PlanData* plan = static_cast<shafft::PlanData*>(plan_ptr);
  return _shafft::getAxes(plan, ca, da, static_cast<shafft::TensorLayout>(layout));
}

// CHANGED: return status, take an out parameter
int shafftGetAllocSize(void* plan_ptr, size_t* alloc_size) {
  if (!plan_ptr || !alloc_size)
    return (int)shafft::Status::SHAFFT_ERR_NULLPTR;
  shafft::PlanData* plan = static_cast<shafft::PlanData*>(plan_ptr);
  return _shafft::getAllocSize(plan, alloc_size);
}

int shafftExecute(void* plan_ptr, shafft_direction_t direction) {
  if (!plan_ptr)
    return (int)shafft::Status::SHAFFT_ERR_NULLPTR;
  shafft::PlanData* plan = static_cast<shafft::PlanData*>(plan_ptr);

  shafft::FFTDirection dir;
  int rc = fftCDirectionCast(direction, &dir);
  if (rc != (int)shafft::Status::SHAFFT_SUCCESS)
    return rc;

  return _shafft::execute(plan, dir);
}

int shafftNormalize(void* plan_ptr) {
  if (!plan_ptr)
    return (int)shafft::Status::SHAFFT_ERR_NULLPTR;
  shafft::PlanData* plan = static_cast<shafft::PlanData*>(plan_ptr);
  return _shafft::normalize(plan);
}

// ---- Portable buffer allocation/free ----------------------------------------

int shafftAllocBufferF(size_t count, void** buf) {
  if (!buf)
    return (int)shafft::Status::SHAFFT_ERR_NULLPTR;
  shafft::complexf* ptr = nullptr;
  int rc = shafft::allocBuffer(count, &ptr);
  *buf = static_cast<void*>(ptr);
  return rc;
}

int shafftAllocBufferD(size_t count, void** buf) {
  if (!buf)
    return (int)shafft::Status::SHAFFT_ERR_NULLPTR;
  shafft::complexd* ptr = nullptr;
  int rc = shafft::allocBuffer(count, &ptr);
  *buf = static_cast<void*>(ptr);
  return rc;
}

int shafftFreeBufferF(void* buf) {
  return shafft::freeBuffer(static_cast<shafft::complexf*>(buf));
}

int shafftFreeBufferD(void* buf) {
  return shafft::freeBuffer(static_cast<shafft::complexd*>(buf));
}

// ---- Portable memory copy ---------------------------------------------------

int shafftCopyToBufferF(void* dst, const void* src, size_t count) {
  return shafft::copyToBuffer(static_cast<shafft::complexf*>(dst),
                              static_cast<const shafft::complexf*>(src), count);
}

int shafftCopyToBufferD(void* dst, const void* src, size_t count) {
  return shafft::copyToBuffer(static_cast<shafft::complexd*>(dst),
                              static_cast<const shafft::complexd*>(src), count);
}

int shafftCopyFromBufferF(void* dst, const void* src, size_t count) {
  return shafft::copyFromBuffer(static_cast<shafft::complexf*>(dst),
                                static_cast<const shafft::complexf*>(src), count);
}

int shafftCopyFromBufferD(void* dst, const void* src, size_t count) {
  return shafft::copyFromBuffer(static_cast<shafft::complexd*>(dst),
                                static_cast<const shafft::complexd*>(src), count);
}

// ---- Library information ----------------------------------------------------

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
