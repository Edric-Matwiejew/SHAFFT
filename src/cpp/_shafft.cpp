#include "_shafft.hpp"

#include <shafft/shafft_error.hpp>
#include <shafft/shafft_types.hpp>

#include "fft_method.h"
#include "normalize.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>  // for fprintf
#include <stdexcept>

using namespace shafft;

static void get_transform_axes(int* coord_spaces, int nca, int* ca, int* nta, int* ta) {
  *nta = 0;
  for (int i = 0; i < nca; i++) {
    if (coord_spaces[ca[i]] == 0) {
      ta[*nta] = ca[i];
      coord_spaces[ca[i]] = 1;
      *nta += 1;
    }
  }
}

namespace _shafft {

#if SHAFFT_BACKEND_HIPFFT
int setStream(shafft::PlanData* plan, hipStream_t stream) noexcept {
  try {
    if (!plan)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);
    if (!plan->subplans)
      SHAFFT_FAIL(SHAFFT_ERR_PLAN_NOT_INIT);
    for (int i = 0; i < plan->nsubplans; ++i) {
      SHAFFT_BACKEND_CHECK(fftSetStream(plan->subplans[i], stream));
    }
    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}
#endif

int setBuffers(shafft::PlanData* plan, void* data, void* work) noexcept;

int getBuffers(shafft::PlanData* plan, void** data, void** work) noexcept;

int planCreate(shafft::PlanData** plan) noexcept {
  if (!plan)
    SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);
  *plan = new (std::nothrow) shafft::PlanData;
  if (!*plan) {
    SHAFFT_FAIL_WITH(SHAFFT_ERR_ALLOC, SHAFFT_ERRSRC_SYSTEM, 0);
  }
  return SHAFFT_STATUS(SHAFFT_SUCCESS);
}

// Plan *planCreate() { return new (std::nothrow) shafft::PlanData; }

// use of subplans is inconsistent, appears here and in the fft methods
// here subplans is the number of high-level transforms stages
// in the fft methods it is the number of low-level transform stages
int planNDA(shafft::PlanData* plan, int ndim, int nda, int dimensions[], shafft::FFTType precision,
            MPI_Comm COMM) noexcept {
  try {
    if (!plan)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);
    if (!dimensions)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);
    if (ndim <= 0)
      SHAFFT_FAIL(SHAFFT_ERR_INVALID_DIM);
    for (int i = 0; i < ndim; ++i)
      if (dimensions[i] <= 0)
        SHAFFT_FAIL(SHAFFT_ERR_INVALID_DIM);
    // nda must leave at least 1 contiguous axis (nca = ndim - nda >= 1)
    if (nda < 0 || nda >= ndim)
      SHAFFT_FAIL(SHAFFT_ERR_INVALID_DECOMP);
    if (COMM == MPI_COMM_NULL)
      SHAFFT_FAIL(SHAFFT_ERR_INVALID_COMM);
    if (precision != FFTType::C2C && precision != FFTType::Z2Z)
      SHAFFT_FAIL(SHAFFT_ERR_INVALID_FFTTYPE);

    // 1D tensors cannot be distributed: require single MPI rank
    int comm_size = 1;
    MPI_Comm_size(COMM, &comm_size);
    if (ndim == 1 && comm_size > 1)
      SHAFFT_FAIL(SHAFFT_ERR_INVALID_DECOMP);  // Cannot distribute a 1D tensor

    // Compute COMM_DIMS: use MPI_Dims_create for balanced distribution, then cap by tensor size
    std::vector<int> COMM_DIMS(ndim, 1);
    if (nda > 0) {
      int nca = ndim - nda;
      // MPI_Dims_create balances factors across axes (e.g., 4 ranks, 2 axes -> [2,2])
      std::vector<int> mpi_dims(nda, 0);
      MPI_Dims_create(comm_size, nda, mpi_dims.data());

      // Cap each distributed axis by both:
      // 1. Its own tensor dimension (dimensions[i])
      // 2. The corresponding contiguous axis it will exchange with (dimensions[i + nca])
      // The cross-axis constraint ensures COMM_DIMS[j] <= dimensions[j + nca] during exchanges
      for (int i = 0; i < nda; ++i) {
        int cap = std::min(dimensions[i], dimensions[i + nca]);
        COMM_DIMS[i] = std::min(mpi_dims[i], cap);
      }
      // Trailing axes remain 1
    }

    // Delegate to planCart with the computed COMM_DIMS
    return planCart(plan, ndim, COMM_DIMS.data(), dimensions, precision, COMM);
  }
  SHAFFT_CATCH_RETURN();
}

int planCart(shafft::PlanData* plan, int ndim, int COMM_DIMS[], int dimensions[],
             shafft::FFTType precision, MPI_Comm COMM) noexcept {
  try {
    if (!plan)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);
    if (!COMM_DIMS || !dimensions)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);
    if (ndim <= 0)
      SHAFFT_FAIL(SHAFFT_ERR_INVALID_DIM);
    if (COMM_DIMS[ndim - 1] != 1)
      SHAFFT_FAIL(SHAFFT_ERR_INVALID_DECOMP);
    for (int i = 0; i < ndim; ++i)
      if (COMM_DIMS[i] < 1 || COMM_DIMS[i] > dimensions[i])
        SHAFFT_FAIL(SHAFFT_ERR_INVALID_DECOMP);

    // Count distributed axes (nda) for validation
    int nda_count = 0;
    for (int i = 0; i < ndim; ++i) {
      if (COMM_DIMS[i] > 1)
        nda_count++;
      else
        break;
    }
    int nca_count = ndim - nda_count;

    // Cross-axis constraint: during exchanges, COMM_DIMS[j] partitions the contiguous
    // axis at position (ndim - nda + j). This must not exceed that axis's dimension.
    // COMM_DIMS[j] <= dimensions[j + nca] for all j in [0, nda)
    for (int j = 0; j < nda_count; ++j) {
      if (COMM_DIMS[j] > dimensions[j + nca_count])
        SHAFFT_FAIL(SHAFFT_ERR_INVALID_DECOMP);
    }

    if (COMM == MPI_COMM_NULL)
      SHAFFT_FAIL(SHAFFT_ERR_INVALID_COMM);
    if (precision != FFTType::C2C && precision != FFTType::Z2Z)
      SHAFFT_FAIL(SHAFFT_ERR_INVALID_FFTTYPE);

    // 1D tensors cannot be distributed: require single MPI rank
    int comm_size = 1;
    MPI_Comm_size(COMM, &comm_size);
    if (ndim == 1 && comm_size > 1)
      SHAFFT_FAIL(SHAFFT_ERR_INVALID_DECOMP);  // Cannot distribute a 1D tensor

    // Element size for host-staging fallback (only used when SHAFFT_GPU_AWARE_MPI=0)
    size_t elem_size = (precision == FFTType::C2C) ? 2 * sizeof(float) : 2 * sizeof(double);

    switch (precision) {
      case FFTType::C2C:
        plan->fft_type = FFTType::C2C;
        plan->slab =
            new (std::nothrow) Slab(ndim, dimensions, COMM_DIMS, MPI_C_COMPLEX, COMM, elem_size);
        break;
      case FFTType::Z2Z:
        plan->fft_type = FFTType::Z2Z;
        plan->slab = new (std::nothrow)
            Slab(ndim, dimensions, COMM_DIMS, MPI_C_DOUBLE_COMPLEX, COMM, elem_size);
        break;
      default:
        SHAFFT_FAIL(SHAFFT_ERR_INVALID_FFTTYPE);
    }
    if (!plan->slab) {
      SHAFFT_FAIL_WITH(SHAFFT_ERR_ALLOC, SHAFFT_ERRSRC_SYSTEM, 0);
    }

    // Handle inactive ranks: no FFT subplans needed, just warn and return success
    if (!plan->slab->is_active()) {
      int world_rank = 0;
      MPI_Comm_rank(COMM, &world_rank);
      std::fprintf(
          stderr,
          "[SHAFFT] Warning: rank %d is inactive and will not participate in FFT computation.\n",
          world_rank);
      plan->nsubplans = 0;
      plan->subplans = nullptr;
      plan->norm_denominator *= std::sqrt(product<int, long double>(dimensions, ndim));
      return SHAFFT_STATUS(SHAFFT_SUCCESS);
    }

    plan->nsubplans = plan->slab->nes() + 1;
    plan->subplans = new (std::nothrow) fftHandle[plan->nsubplans]();  // value-init to null members
    if (!plan->subplans) {
      SHAFFT_FAIL_WITH(SHAFFT_ERR_ALLOC, SHAFFT_ERRSRC_SYSTEM, 0);
    }

    int subsize[ndim], offset[ndim];
    const int nca = plan->slab->nca();
    int ca[nca], ta[ndim], nta;
    int coord_spaces[ndim];
    for (int i = 0; i < ndim; ++i)
      coord_spaces[i] = 0;

    for (int i = 0; i < plan->nsubplans; ++i) {
      int cfg_rc = plan->slab->get_ith_config(subsize, offset, ca, i);
      if (cfg_rc != 0)
        return cfg_rc;
      get_transform_axes(coord_spaces, nca, ca, &nta, ta);
      SHAFFT_BACKEND_CHECK(fftPlan(plan->subplans[i], nta, ta, ndim, subsize, plan->fft_type));
    }

    plan->norm_denominator *= std::sqrt(product<int, long double>(dimensions, ndim));
    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

int destroy(shafft::PlanData** plan) noexcept {
  try {
    if (!plan || !*plan)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);
    shafft::PlanData* p = *plan;

    if (p->subplans) {
      for (int i = 0; i < p->nsubplans; ++i) {
        // best-effort destroy; record first failure
        int rc = fftDestroy(p->subplans[i]);
        if (rc != 0) {
#if SHAFFT_BACKEND_HIPFFT
          shafft::detail::set_last_error(shafft::Status::SHAFFT_ERR_BACKEND, SHAFFT_ERRSRC_HIPFFT,
                                         rc);
#else
          shafft::detail::set_last_error(shafft::Status::SHAFFT_ERR_BACKEND, SHAFFT_ERRSRC_FFTW,
                                         rc);
#endif
        }
      }
      delete[] p->subplans;
      p->subplans = nullptr;
      p->nsubplans = 0;
    }

    delete p->slab;
    p->slab = nullptr;

    p->norm_exponent = 0;
    p->norm_denominator = 1.0L;

    delete p;
    *plan = nullptr;

    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

int getLayout(PlanData* plan, int* subsize, int* offset, shafft::TensorLayout layout) noexcept {
  try {
    if (!plan || !plan->slab)
      SHAFFT_FAIL(SHAFFT_ERR_PLAN_NOT_INIT);
    if (!subsize || !offset)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);

    Slab* s = plan->slab;
    int rc = 0;
    switch (layout) {
      case TensorLayout::CURRENT:
        s->get_subsize(subsize);
        s->get_offset(offset);
        break;
      case TensorLayout::INITIAL:
        rc = s->get_ith_layout(subsize, offset, 0);
        break;
      case TensorLayout::TRANSFORMED:
        rc = s->get_ith_layout(subsize, offset, s->nes());
        break;
    }
    if (rc != 0)
      return rc;
    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

int getAxes(PlanData* plan, int* ca, int* da, shafft::TensorLayout layout) noexcept {
  try {
    if (!plan || !plan->slab)
      SHAFFT_FAIL(SHAFFT_ERR_PLAN_NOT_INIT);
    if (!ca || !da)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);

    Slab* s = plan->slab;
    int rc = 0;
    switch (layout) {
      case TensorLayout::CURRENT:
        s->get_ca(ca);
        s->get_da(da);
        break;
      case TensorLayout::INITIAL:
        rc = s->get_ith_axes(ca, da, 0);
        break;
      case TensorLayout::TRANSFORMED:
        rc = s->get_ith_axes(ca, da, s->nes());
        break;
    }
    if (rc != 0)
      return rc;
    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

int getAllocSize(shafft::PlanData* plan, size_t* alloc_size) noexcept {
  try {
    if (!alloc_size)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);
    if (!plan || !plan->slab)
      SHAFFT_FAIL(SHAFFT_ERR_PLAN_NOT_INIT);
    *alloc_size = plan->slab->alloc_size();
    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

int execute(shafft::PlanData* plan, shafft::FFTDirection direction) noexcept {
  try {
    if (!plan)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);
    if (!plan->slab)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);

    // Inactive ranks have no subplans; silently succeed
    if (!plan->slab->is_active()) {
      plan->norm_exponent += 1;
      return SHAFFT_STATUS(SHAFFT_SUCCESS);
    }

    if (!plan->subplans)
      SHAFFT_FAIL(SHAFFT_ERR_PLAN_NOT_INIT);

    const int nsp = plan->nsubplans;
    if (nsp < 1)
      SHAFFT_FAIL(SHAFFT_ERR_PLAN_NOT_INIT);

    void *data = nullptr, *work = nullptr;
    for (int i = 0; i < nsp; ++i) {
      const int subindex = (direction == FFTDirection::FORWARD) ? i : (nsp - 1 - i);

      plan->slab->get_buffers(&data, &work);
      if (!data || !work)
        SHAFFT_FAIL(SHAFFT_ERR_NO_BUFFER);

      SHAFFT_BACKEND_CHECK(fftExecute(plan->subplans[subindex], data, work, direction));

      // fftExecute swaps pointers internally so result is always in 'data'
      // Update slab's buffer pointers to match
      plan->slab->set_buffers(data, work);

      if ((subindex < nsp - 1) && (direction == FFTDirection::FORWARD)) {
        int rc = plan->slab->forward();
        if (rc != 0)
          return rc;
      }
      if ((subindex > 0) && (direction == FFTDirection::BACKWARD)) {
        int rc = plan->slab->backward();
        if (rc != 0)
          return rc;
      }
    }

    plan->current_layout = (plan->current_layout == TensorLayout::INITIAL) ? TensorLayout::CURRENT
                                                                           : TensorLayout::INITIAL;

    plan->norm_exponent += 1;
    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

int normalize(shafft::PlanData* plan) noexcept {
  try {
    if (!plan)
      SHAFFT_FAIL(SHAFFT_ERR_NULLPTR);
    if (!plan->slab)
      SHAFFT_FAIL(SHAFFT_ERR_PLAN_NOT_INIT);

    // Inactive ranks have no data to normalize; silently succeed
    if (!plan->slab->is_active()) {
      plan->norm_exponent = 0;
      return SHAFFT_STATUS(SHAFFT_SUCCESS);
    }

    int subsize[plan->slab->ndim()];
    plan->slab->get_subsize(subsize);
    long double norm_factor = 1.0 / std::pow(plan->norm_denominator, plan->norm_exponent);

    void *data = nullptr, *work = nullptr;
    plan->slab->get_buffers(&data, &work);
    if (!data || !work)
      SHAFFT_FAIL(SHAFFT_ERR_NO_BUFFER);

    const size_t tensor_size = product<int, size_t>(subsize, plan->slab->ndim());
    int norm_rc = 0;

#if SHAFFT_BACKEND_HIPFFT
    // Get stream from first subplan for async kernel execution
    hipStream_t stream = nullptr;
    if (plan->subplans && plan->nsubplans > 0) {
      stream = plan->subplans[0].stream;
    }
    switch (plan->fft_type) {
      case FFTType::C2C:
        norm_rc = normalizeComplexFloat((float)norm_factor, tensor_size, data, stream);
        break;
      case FFTType::Z2Z:
        norm_rc = normalizeComplexDouble((double)norm_factor, tensor_size, data, stream);
        break;
    }
    // Sync stream to ensure normalize is complete before returning
    // (maintains same contract as fftExecute - operations complete on return)
    if (norm_rc == 0) {
      hipError_t sync_err = hipStreamSynchronize(stream);
      if (sync_err != hipSuccess) {
        SHAFFT_FAIL_WITH(SHAFFT_ERR_BACKEND, SHAFFT_ERRSRC_HIP, static_cast<int>(sync_err));
      }
    }
#else
    switch (plan->fft_type) {
      case FFTType::C2C:
        norm_rc = normalizeComplexFloat((float)norm_factor, tensor_size, data);
        break;
      case FFTType::Z2Z:
        norm_rc = normalizeComplexDouble((double)norm_factor, tensor_size, data);
        break;
    }
#endif

    if (norm_rc != 0) {
#if SHAFFT_BACKEND_HIPFFT
      SHAFFT_FAIL_WITH(SHAFFT_ERR_BACKEND, SHAFFT_ERRSRC_HIP, norm_rc);
#else
      SHAFFT_FAIL_WITH(SHAFFT_ERR_BACKEND, SHAFFT_ERRSRC_FFTW, norm_rc);
#endif
    }
    plan->norm_exponent = 0;
    return SHAFFT_STATUS(SHAFFT_SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

}  // namespace _shafft
