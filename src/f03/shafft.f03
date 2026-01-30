!> @file shafft.f03
!! @brief Fortran 2003 interface to SHAFFT.
!!
!! Conventions:
!! - Axis indices are **0-based** (values returned in arrays like `ca` follow C indexing).
!! - Arrays noted as "length = ndim" must have at least `ndim` elements.
!! - **ca** = indices of locally contiguous (non-distributed) axes, ordered
!!   **innermost to outermost stride** for the reported stage/layout.
!! - Buffers may be swapped internally during execution; call `shafftGetBuffers`
!!   **after** `shafftExecute` to obtain the buffer that currently holds the
!!   transformed data.
module shafft
   use iso_c_binding
#include "shafft_config.h"

   implicit none

   private

   ! FFT type constants
   public :: SHAFFT_C2C, SHAFFT_Z2Z

   ! Transform direction constants
   public :: SHAFFT_FORWARD, SHAFFT_BACKWARD

   ! Tensor layout constants
   public :: SHAFFT_TENSOR_LAYOUT_CURRENT
   public :: SHAFFT_TENSOR_LAYOUT_INITIAL
   public :: SHAFFT_TENSOR_LAYOUT_TRANSFORMED

   ! Configuration subroutines
   public :: shafftConfigurationNDA, shafftConfigurationCart

   ! Plan creation and destruction
   public :: shafftPlan, shafftPlanNDA, shafftPlanCart
   public :: shafftDestroy

   ! Plan execution
   public :: shafftExecute, shafftNormalize

   ! Buffer management (generic interfaces)
   public :: shafftGetAllocSize
   public :: shafftSetBuffers, shafftGetBuffers
   public :: shafftAllocBuffer, shafftFreeBuffer
   public :: shafftCopyToBuffer, shafftCopyFromBuffer

   ! Buffer management (precision-specific, also accessible via generic interfaces)
   public :: shafftSetBuffers_sp, shafftSetBuffers_dp
   public :: shafftGetBuffers_sp, shafftGetBuffers_dp
   public :: shafftAllocBuffer_sp, shafftAllocBuffer_dp
   public :: shafftFreeBuffer_sp, shafftFreeBuffer_dp
   public :: shafftCopyToBuffer_sp, shafftCopyToBuffer_dp
   public :: shafftCopyFromBuffer_sp, shafftCopyFromBuffer_dp

   ! Layout and axis queries
   public :: shafftGetLayout, shafftGetAxes

   ! Error query functions
   public :: shafftLastErrorStatus, shafftLastErrorSource, shafftLastErrorDomainCode
   public :: shafftLastErrorMessage, shafftClearLastError, shafftErrorSourceName

   ! Library information
   public :: shafftGetBackendName, shafftGetVersion, shafftGetVersionString

   ! FFT Type Constants
   integer(c_int), parameter :: SHAFFT_C2C = 0  ! Single-precision complex-to-complex
   integer(c_int), parameter :: SHAFFT_Z2Z = 1  ! Double-precision complex-to-complex

   ! Transform Direction Constants
   integer(c_int), parameter :: SHAFFT_FORWARD  = 0  ! Forward FFT
   integer(c_int), parameter :: SHAFFT_BACKWARD = 1  ! Backward (inverse) FFT

   ! Tensor Layout Constants
   integer, parameter :: SHAFFT_TENSOR_LAYOUT_CURRENT     = 0  ! Current layout
   integer, parameter :: SHAFFT_TENSOR_LAYOUT_INITIAL     = 1  ! Initial layout (before transforms)
   integer, parameter :: SHAFFT_TENSOR_LAYOUT_TRANSFORMED = 2  ! Transformed layout (after transforms)

   ! Error Source Constants
   integer, parameter, public :: SHAFFT_ERRSRC_NONE   = 0  ! No error or SHAFFT-internal error
   integer, parameter, public :: SHAFFT_ERRSRC_MPI    = 1  ! MPI library error
   integer, parameter, public :: SHAFFT_ERRSRC_HIP    = 2  ! HIP runtime error
   integer, parameter, public :: SHAFFT_ERRSRC_HIPFFT = 3  ! hipFFT library error
   integer, parameter, public :: SHAFFT_ERRSRC_FFTW   = 4  ! FFTW library error
   integer, parameter, public :: SHAFFT_ERRSRC_SYSTEM = 5  ! OS / allocation errors


   interface

      function c_shafftConfigurationNDA(ndim, size, nda, subsize, offset, COMM_DIMS, &
                                         precision, mem_limit, comm) bind(C, name="shafftConfigurationNDAf03") result(status)
         use iso_c_binding
         integer(c_int), value :: ndim
         type(c_ptr), value :: size
         integer(c_int) :: nda
         type(c_ptr), value :: subsize
         type(c_ptr), value :: offset
         type(c_ptr), value :: COMM_DIMS
         integer(kind(SHAFFT_C2C)), value :: precision
         integer(c_size_t), value :: mem_limit
         integer, intent(in) :: comm
         integer(c_int) :: status
      end function c_shafftConfigurationNDA

      function c_shafftConfigurationCart(ndim, size, subsize, offset, COMM_DIMS, COMM_SIZE, &
                                               precision, mem_limit, comm) &
         bind(C, name="shafftConfigurationCartf03") result(status)
         use iso_c_binding
         integer(c_int), value :: ndim
         type(c_ptr), value :: size
         type(c_ptr), value :: subsize
         type(c_ptr), value :: offset
         type(c_ptr), value :: COMM_DIMS
         integer(c_int) :: COMM_SIZE
         integer(kind(SHAFFT_C2C)), value :: precision
         integer(c_size_t), value :: mem_limit
         integer, intent(in) :: comm
         integer(c_int) :: status
      end function c_shafftConfigurationCart

      function c_shafftPlanCreate(out_plan) bind(C, name="shafftPlanCreate") result(status)
         use iso_c_binding
         type(c_ptr) :: out_plan
         integer(c_int) :: status
      end function c_shafftPlanCreate

      function c_shafftPlanNDA(plan, ndim, nda, dimensions, &
                                precision, comm) bind(C, name="shafftPlanNDAf03") result(status)
         use iso_c_binding
         type(c_ptr), value :: plan
         integer(c_int), value :: ndim
         integer(c_int), value :: nda
         type(c_ptr), value :: dimensions
         integer(kind(SHAFFT_C2C)), value :: precision
         integer, intent(in) :: comm
         integer(c_int) :: status
      end function c_shafftPlanNDA

      function c_shafftPlanCart(plan, ndim, COMM_DIMS, dimensions, &
                                      precision, comm) bind(C, name="shafftPlanCartf03") result(status)
         use iso_c_binding
         type(c_ptr), value :: plan
         integer(c_int), value :: ndim
         type(c_ptr), value :: COMM_DIMS
         type(c_ptr), value :: dimensions
         integer(kind(SHAFFT_C2C)), value :: precision
         integer, intent(in) :: comm
         integer(c_int) :: status
      end function c_shafftPlanCart

      function c_shafftGetAllocSize(plan, alloc_size) bind(C, name="shafftGetAllocSize") result(status)
         use iso_c_binding
         type(c_ptr), value :: plan
         integer(c_size_t) :: alloc_size
         integer(c_int) :: status
      end function c_shafftGetAllocSize

      function c_shafftExecute(plan, direction) bind(C, name="shafftExecute") result(status)
         use iso_c_binding
         type(c_ptr), value :: plan
         integer(c_int), value :: direction
         integer(c_int) :: status
      end function c_shafftExecute

      function c_shafftNormalize(plan) bind(c, name="shafftNormalize") result(status)
         use iso_c_binding
         type(c_ptr), value :: plan
         integer(c_int) :: status
      end function c_shafftNormalize

      function c_shafftSetBuffers(plan, data, work) bind(c, name="shafftSetBuffers") result(status)
         use iso_c_binding
         type(c_ptr), value :: plan
         type(c_ptr), value :: data
         type(c_ptr), value :: work
         integer(c_int) :: status
      end function c_shafftSetBuffers

      function c_shafftGetBuffers(plan, data, work) bind(c, name="shafftGetBuffers") result(status)
         use iso_c_binding
         type(c_ptr), value :: plan
         type(c_ptr) :: data
         type(c_ptr) :: work
         integer(c_int) :: status
      end function c_shafftGetBuffers

      function c_shafftDestroy(plan) bind(C, name="shafftDestroy") result(status)
         use iso_c_binding
         type(c_ptr), intent(inout) :: plan
         integer(c_int) :: status
      end function c_shafftDestroy

      function c_shafftGetLayout(plan, subsize, offset, layout) bind(C, name="shafftGetLayout") result(status)
         use iso_c_binding
         type(c_ptr), value :: plan
         type(c_ptr), value :: subsize
         type(c_ptr), value :: offset
         integer(c_int), value :: layout
         integer(c_int) :: status
      end function c_shafftGetLayout

      function c_shafftGetAxes(plan, ca, da, layout) bind(C, name="shafftGetAxes") result(status)
         use iso_c_binding
         type(c_ptr), value :: plan
         type(c_ptr), value :: ca
         type(c_ptr), value :: da
         integer(c_int), value :: layout
         integer(c_int) :: status
      end function c_shafftGetAxes

      function c_shafftSetStream(plan, stream) bind(C, name="shafftSetStream") result(status)
         use iso_c_binding
         type(c_ptr), value :: plan
         type(c_ptr), value :: stream
         integer(c_int) :: status
      end function c_shafftSetStream

      ! ---- Portable buffer allocation/free --------------------------------

      function c_shafftAllocBufferF(count, buf) bind(C, name="shafftAllocBufferF") result(status)
         use iso_c_binding
         integer(c_size_t), value :: count
         type(c_ptr) :: buf
         integer(c_int) :: status
      end function c_shafftAllocBufferF

      function c_shafftAllocBufferD(count, buf) bind(C, name="shafftAllocBufferD") result(status)
         use iso_c_binding
         integer(c_size_t), value :: count
         type(c_ptr) :: buf
         integer(c_int) :: status
      end function c_shafftAllocBufferD

      function c_shafftFreeBufferF(buf) bind(C, name="shafftFreeBufferF") result(status)
         use iso_c_binding
         type(c_ptr), value :: buf
         integer(c_int) :: status
      end function c_shafftFreeBufferF

      function c_shafftFreeBufferD(buf) bind(C, name="shafftFreeBufferD") result(status)
         use iso_c_binding
         type(c_ptr), value :: buf
         integer(c_int) :: status
      end function c_shafftFreeBufferD

      ! ---- Portable memory copy -------------------------------------------

      function c_shafftCopyToBufferF(dst, src, count) bind(C, name="shafftCopyToBufferF") result(status)
         use iso_c_binding
         type(c_ptr), value :: dst
         type(c_ptr), value :: src
         integer(c_size_t), value :: count
         integer(c_int) :: status
      end function c_shafftCopyToBufferF

      function c_shafftCopyToBufferD(dst, src, count) bind(C, name="shafftCopyToBufferD") result(status)
         use iso_c_binding
         type(c_ptr), value :: dst
         type(c_ptr), value :: src
         integer(c_size_t), value :: count
         integer(c_int) :: status
      end function c_shafftCopyToBufferD

      function c_shafftCopyFromBufferF(dst, src, count) bind(C, name="shafftCopyFromBufferF") result(status)
         use iso_c_binding
         type(c_ptr), value :: dst
         type(c_ptr), value :: src
         integer(c_size_t), value :: count
         integer(c_int) :: status
      end function c_shafftCopyFromBufferF

      function c_shafftCopyFromBufferD(dst, src, count) bind(C, name="shafftCopyFromBufferD") result(status)
         use iso_c_binding
         type(c_ptr), value :: dst
         type(c_ptr), value :: src
         integer(c_size_t), value :: count
         integer(c_int) :: status
      end function c_shafftCopyFromBufferD

      ! ---- Library information --------------------------------------------

      function c_shafftGetBackendName() bind(C, name="shafftGetBackendName") result(name)
         use iso_c_binding
         type(c_ptr) :: name
      end function c_shafftGetBackendName

      subroutine c_shafftGetVersion(major, minor, patch) bind(C, name="shafftGetVersion")
         use iso_c_binding
         integer(c_int) :: major
         integer(c_int) :: minor
         integer(c_int) :: patch
      end subroutine c_shafftGetVersion

      function c_shafftGetVersionString() bind(C, name="shafftGetVersionString") result(version)
         use iso_c_binding
         type(c_ptr) :: version
      end function c_shafftGetVersionString

      ! ---- Error query functions ------------------------------------------

      function c_shafft_last_error_status() bind(C, name="shafft_last_error_status") result(status)
         use iso_c_binding
         integer(c_int) :: status
      end function c_shafft_last_error_status

      function c_shafft_last_error_source() bind(C, name="shafft_last_error_source") result(source)
         use iso_c_binding
         integer(c_int) :: source
      end function c_shafft_last_error_source

      function c_shafft_last_error_domain_code() bind(C, name="shafft_last_error_domain_code") result(code)
         use iso_c_binding
         integer(c_int) :: code
      end function c_shafft_last_error_domain_code

      function c_shafft_last_error_message(buf, buflen) bind(C, name="shafft_last_error_message") result(length)
         use iso_c_binding
         type(c_ptr), value :: buf
         integer(c_int), value :: buflen
         integer(c_int) :: length
      end function c_shafft_last_error_message

      subroutine c_shafft_clear_last_error() bind(C, name="shafft_clear_last_error")
         use iso_c_binding
      end subroutine c_shafft_clear_last_error

      function c_shafft_error_source_name(source) bind(C, name="shafft_error_source_name") result(name)
         use iso_c_binding
         integer(c_int), value :: source
         type(c_ptr) :: name
      end function c_shafft_error_source_name
         
   end interface

   ! Generic interface: dispatches to shafftPlanNDA or shafftPlanCart
   interface shafftPlan
      module procedure shafftPlanNDA
      module procedure shafftPlanCart
   end interface shafftPlan

   ! Generic interface: dispatches to shafftSetBuffers_sp or shafftSetBuffers_dp
   interface shafftSetBuffers
      module procedure shafftSetBuffers_sp
      module procedure shafftSetBuffers_dp
   end interface shafftSetBuffers

   ! Generic interface: dispatches to shafftGetBuffers_sp or shafftGetBuffers_dp
   interface shafftGetBuffers
      module procedure shafftGetBuffers_sp
      module procedure shafftGetBuffers_dp
   end interface shafftGetBuffers

   ! Generic interface: dispatches to shafftAllocBuffer_sp or shafftAllocBuffer_dp
   interface shafftAllocBuffer
      module procedure shafftAllocBuffer_sp
      module procedure shafftAllocBuffer_dp
   end interface shafftAllocBuffer

   ! Generic interface: dispatches to shafftFreeBuffer_sp or shafftFreeBuffer_dp
   interface shafftFreeBuffer
      module procedure shafftFreeBuffer_sp
      module procedure shafftFreeBuffer_dp
   end interface shafftFreeBuffer

   ! Generic interface: dispatches to shafftCopyToBuffer_sp or shafftCopyToBuffer_dp
   interface shafftCopyToBuffer
      module procedure shafftCopyToBuffer_sp
      module procedure shafftCopyToBuffer_dp
   end interface shafftCopyToBuffer

   ! Generic interface: dispatches to shafftCopyFromBuffer_sp or shafftCopyFromBuffer_dp
   interface shafftCopyFromBuffer
      module procedure shafftCopyFromBuffer_sp
      module procedure shafftCopyFromBuffer_dp
   end interface shafftCopyFromBuffer

contains

   subroutine shafftCheck(status)
      integer(c_int), intent(in) :: status
      if (status /= 0) then
         write (*, *) "Error executing shafft plan with status: ", status
         stop
      end if
   end subroutine shafftCheck

   !> @brief Compute an N-D slab decomposition with a specified number of distributed axes.
   !! @ingroup fortran_api
   !! @details
   !! Determines the optimal Cartesian process grid and local tensor block for each
   !! MPI rank based on the global tensor size, desired decomposition, and optional
   !! memory constraints. Use the outputs to call `shafftPlanNDA`.
   !!
   !! **Auto-selection mode (nda == 0 on input):**
   !! When `nda` is 0, the planner automatically selects the number of distributed axes.
   !! The `mem_limit` parameter controls the selection strategy:
   !! - `mem_limit > 0`: Maximize nda subject to per-rank memory staying under limit
   !! - `mem_limit == 0`: Maximize nda (no memory constraint)
   !! - `mem_limit < 0` (signed interpretation): Minimize nda (fewest distributed axes)
   !!
   !! **Manual mode (nda > 0 on input):**
   !! When `nda` is positive, that exact value is used. The function fails if the
   !! requested decomposition cannot be satisfied.
   !!
   !! **Process grid computation:**
   !! The Cartesian process grid `COMM_DIMS` is computed automatically. The grid follows
   !! a "slab prefix" structure: the first `nda` entries may be > 1, trailing entries are 1.
   !! For example, with 8 ranks on a 64x64x32 tensor, COMM_DIMS might be [2,4,1].
   !!
   !! **Per-axis caps:**
   !! Each COMM_DIMS(i) is capped by min(size(i), size(ndim-i+1)) to ensure valid
   !! redistribution during the FFT computation.
   !!
   !! **Inactive ranks:**
   !! If the tensor cannot be evenly distributed, some ranks may become inactive.
   !! Inactive ranks are handled gracefully: plan creation succeeds, execute()
   !! and normalize() become no-ops, and a warning is printed to stderr.
   !!
   !! @param[in]  ndim       Number of tensor dimensions.
   !! @param[in]  size       Global extents per axis (length = ndim).
   !! @param[inout] nda      Desired distributed axes on input (0 for auto);
   !!                        actual value chosen by the planner on output.
   !! @param[out] subsize    Local extents per axis for this rank (length = ndim).
   !! @param[out] offset     Global starting indices per axis for this rank (length = ndim).
   !! @param[out] COMM_DIMS  Cartesian process-grid dims (length = ndim).
   !!                        Leading `nda` entries contain the grid; trailing entries are 1.
   !! @param[in]  precision  `SHAFFT_C2C` (single) or `SHAFFT_Z2Z` (double).
   !! @param[in]  mem_limit  Per-rank memory limit in bytes (see auto-selection above).
   !! @param[in]  comm       MPI communicator.
   subroutine shafftConfigurationNDA(ndim, size, nda, subsize, offset, COMM_DIMS, precision, mem_limit, comm)
      integer(c_int), intent(in) :: ndim
      integer(c_int), dimension(ndim), target, intent(in) :: size
      integer(c_int), intent(in) :: nda
      integer(c_int), dimension(ndim), target, intent(out) :: subsize
      integer(c_int), dimension(ndim), target, intent(out) :: offset
      integer(c_int), dimension(ndim), target, intent(out) :: COMM_DIMS
      integer(kind(SHAFFT_C2C)), intent(in) :: precision
      integer(c_size_t), intent(in) :: mem_limit
      integer(c_int), intent(in) :: comm
      call shafftCheck(c_shafftConfigurationNDA(ndim, c_loc(size), nda, c_loc(subsize), c_loc(offset), &
                                                 c_loc(COMM_DIMS), precision, mem_limit, comm))
   end subroutine shafftConfigurationNDA

   !> @brief Compute a Cartesian decomposition and report communicator size.
   !! @ingroup fortran_api
   !! @details
   !! Either validates a user-provided Cartesian process grid or auto-selects one,
   !! then computes the local tensor block for each rank. Use the outputs to call
   !! `shafftPlanCart`.
   !!
   !! **Auto-selection mode (COMM_DIMS all zeros on input):**
   !! When all entries of `COMM_DIMS` are 0, the planner automatically selects
   !! the optimal grid. The `mem_limit` parameter controls the strategy:
   !! - `mem_limit >= 0`: Maximize number of distributed axes
   !! - `mem_limit < 0`: Minimize number of distributed axes
   !!
   !! **Manual mode (COMM_DIMS non-zero on input):**
   !! When `COMM_DIMS` contains non-zero values, the provided grid is validated
   !! and used directly. The grid must follow the "slab prefix" structure:
   !! - Leading entries (indices 1..d) must be > 1
   !! - Trailing entries (indices d+1..ndim) must be 1 (or 0, normalized to 1)
   !! - No gaps allowed (e.g., [2,1,4] is invalid)
   !!
   !! **Grid constraints:**
   !! - Each COMM_DIMS(i) must not exceed min(size(i), size(ndim-i+1))
   !! - The product of COMM_DIMS must not exceed the number of MPI ranks
   !! - Single rank: COMM_DIMS must be all 1s
   !!
   !! **COMM_SIZE output:**
   !! Set to the product of leading COMM_DIMS entries where COMM_DIMS(i) > 1.
   !! This is the number of ranks that participate; remaining ranks are inactive.
   !!
   !! **Inactive ranks:**
   !! Ranks with world_rank >= COMM_SIZE do not participate in the computation.
   !! They are handled gracefully: plan creation succeeds, execute() and
   !! normalize() become no-ops, and a warning is printed to stderr.
   !!
   !! @param[in]  ndim       Number of tensor dimensions.
   !! @param[in]  size       Global extents per axis (length = ndim).
   !! @param[out] subsize    Local extents per axis for this rank (length = ndim).
   !! @param[out] offset     Global starting indices per axis for this rank (length = ndim).
   !! @param[inout] COMM_DIMS Cartesian process-grid dims (length = ndim).
   !!                        On input: zeros for auto-select, or explicit grid.
   !!                        On output: validated/chosen grid with trailing 1s.
   !! @param[out] COMM_SIZE  Number of active ranks (product of leading grid dims).
   !! @param[in]  precision  `SHAFFT_C2C` (single) or `SHAFFT_Z2Z` (double).
   !! @param[in]  mem_limit  Per-rank memory limit in bytes.
   !! @param[in]  comm       MPI communicator.
   subroutine shafftConfigurationCart( ndim, size, subsize, offset, &
                                             COMM_DIMS, COMM_SIZE, precision, mem_limit, comm)
      integer(c_int), intent(in) :: ndim
      integer(c_int), target, dimension(ndim), intent(in) :: size
      integer(c_int), target, dimension(ndim), intent(out) :: subsize
      integer(c_int), target, dimension(ndim), intent(out) :: offset
      integer(c_int), target, dimension(ndim), intent(out) :: COMM_DIMS
      integer(c_int), intent(out) :: COMM_SIZE
      integer(kind(SHAFFT_C2C)), intent(in) :: precision
      integer(c_size_t), intent(in) :: mem_limit
      integer(c_int), intent(in) :: comm
      call shafftCheck(c_shafftConfigurationCart(  ndim, c_loc(size), c_loc(subsize), c_loc(offset), &
                                                         c_loc(COMM_DIMS), COMM_SIZE, &
                                                         precision, mem_limit, comm))
   end subroutine shafftConfigurationCart

   !> @brief Build a plan from an NDA decomposition.
   !! @ingroup fortran_api
   !! @details
   !! Also accessible via the generic interface `shafftPlan`.
   !! See `shafftConfigurationNDA` for decomposition setup.
   !! @param[out] plan       Plan pointer .
   !! @param[in]  ndim       Global rank (number of axes).
   !! @param[in]  nda        Number of distributed axes (NDA rank).
   !! @param[in]  dimensions Global extents per axis (length = ndim).
   !! @param[in]  precision  `SHAFFT_C2C` or `SHAFFT_Z2Z`.
   !! @param[in]  comm       MPI communicator (Fortran handle).
   !! @see shafftPlan, shafftPlanCart
   subroutine shafftPlanNDA(plan, ndim, nda, dimensions, precision, comm)
      integer(c_int), intent(in) :: ndim 
      type(c_ptr), intent(out) :: plan
      integer(c_int), intent(in) :: nda
      integer(c_int), target, intent(in) :: dimensions(ndim)
      integer(kind(SHAFFT_C2C)), intent(in) :: precision
      integer(c_int), intent(in) :: comm
      call shafftCheck(c_shafftPlanCreate(plan))
      call shafftCheck(c_shafftPlanNDA(plan, ndim, nda, c_loc(dimensions), precision, comm))
   end subroutine shafftPlanNDA

   !> @brief Build a plan from an explicit Cartesian process grid.
   !! @ingroup fortran_api
   !! @details
   !! Also accessible via the generic interface `shafftPlan`.
   !! See `shafftConfigurationCart` for decomposition setup.
   !! @param[out] plan       Plan pointer .
   !! @param[in]  COMM_DIMS  Cartesian process-grid dims (length = number of distributed axes).
   !! @param[in]  dimensions Global extents per axis (length = size(dimensions)).
   !! @param[in]  precision  `SHAFFT_C2C` or `SHAFFT_Z2Z`.
   !! @param[in]  comm       MPI communicator (Fortran handle).
   !! @see shafftPlan, shafftPlanNDA
   subroutine shafftPlanCart(plan, COMM_DIMS, dimensions, precision, comm)
      type(c_ptr), intent(out) :: plan
      integer(c_int), target, intent(in) :: COMM_DIMS(:)
      integer(c_int), target, intent(in) :: dimensions(:)
      integer(kind(SHAFFT_C2C)), intent(in) :: precision
      integer(c_int), intent(in) :: comm
      integer(c_int) :: ndim
      ndim = size(dimensions)
      call shafftCheck(c_shafftPlanCreate(plan))
      call shafftCheck(c_shafftPlanCart(plan, ndim, c_loc(COMM_DIMS), c_loc(dimensions), precision, comm))
   end subroutine shafftPlanCart


   !> @brief Execute the FFT associated with the plan.
   !! @ingroup fortran_api
   !! @details
   !! Performs the forward or backward transform on the attached buffers according to the plan.
   !! @param[in,out] plan     Plan pointer .
   !! @param[in]     direction `SHAFFT_FORWARD` or `SHAFFT_BACKWARD`.
   subroutine shafftExecute(plan, direction)
      type(c_ptr), intent(inout) :: plan
      integer(c_int), intent(in) :: direction
      call shafftCheck(c_shafftExecute(plan, direction))
   end subroutine shafftExecute


   !> @brief Destroy a plan and release its resources.
   !! @ingroup fortran_api
   !! @param[in,out] plan  Plan pointer .
   !! On return, plan is set to c_null_ptr.
   subroutine shafftDestroy(plan)
      use iso_c_binding
      type(c_ptr), intent(inout) :: plan
      call shafftCheck(c_shafftDestroy(plan))
   end subroutine shafftDestroy

   !> @brief Apply the library's normalization to the current data buffer.
   !! @ingroup fortran_api
   !! @param[in,out] plan  Plan pointer.
   subroutine shafftNormalize(plan)
      type(c_ptr), intent(inout) :: plan
      call shafftCheck(c_shafftNormalize(plan))
   end subroutine shafftNormalize


   !> @brief Report the total buffer size required by the plan (in elements).
   !! @ingroup fortran_api
   !! @param[in]  plan        Plan pointer.
   !! @param[out] alloc_size  Required element count; 0 on error.
   subroutine shafftGetAllocSize(plan, alloc_size)
      type(c_ptr), intent(in) :: plan
      integer(c_size_t), intent(out) :: alloc_size
      call shafftCheck(c_shafftGetAllocSize(plan, alloc_size))
   end subroutine shafftGetAllocSize


   !> @brief Attach data and work buffers to a plan (single precision).
   !! @ingroup fortran_api
   !! @details
   !! The plan will use these buffers for redistributions and FFT kernels.
   !! The caller allocates and owns the buffers. Buffers must remain valid
   !! for the lifetime of the plan or until new buffers are set.
   !! @param[in,out] plan   Plan pointer .
   !! @param[in,out] fdata  Complex(single) data array pointer.
   !! @param[in,out] fwork  Complex(single) work/scratch array pointer.
   subroutine shafftSetBuffers_sp(plan, fdata, fwork)
      type(c_ptr), intent(inout) :: plan
      complex(c_float), pointer, intent(inout) :: fdata(:)
      complex(c_float), pointer, intent(inout) :: fwork(:)
      call shafftCheck(c_shafftSetBuffers(plan, c_loc(fdata), c_loc(fwork)))
   end subroutine shafftSetBuffers_sp
   

   !> @brief Attach data and work buffers to a plan (double precision).
   !! @ingroup fortran_api
   !! @details
   !! The plan will use these buffers for redistributions and FFT kernels.
   !! The caller allocates and owns the buffers. Buffers must remain valid
   !! for the lifetime of the plan or until new buffers are set.
   !! @param[in,out] plan   Plan pointer.
   !! @param[in,out] fdata  Complex(double) data array pointer.
   !! @param[in,out] fwork  Complex(double) work/scratch array pointer.
   subroutine shafftSetBuffers_dp(plan, fdata, fwork)
      type(c_ptr), intent(inout) :: plan
      complex(c_double), pointer, intent(inout) :: fdata(:)
      complex(c_double), pointer, intent(inout) :: fwork(:)
      call shafftCheck(c_shafftSetBuffers(plan, c_loc(fdata), c_loc(fwork)))
   end subroutine shafftSetBuffers_dp
   
   !> @brief Retrieve the currently attached buffers (single precision).
   !! @ingroup fortran_api
   !! @details
   !! **Important:** During execution the library may swap the roles of `data` and `work`.
   !! Call this **after** `shafftExecute` to locate the buffer that holds the transformed data.
   !! @param[in]  plan       Plan pointer.
   !! @param[in]  alloc_size Number of complex elements (from shafftGetAllocSize).
   !! @param[out] data       Complex(single) data array pointer (associated on return).
   !! @param[out] work       Complex(single) work array pointer (associated on return).
   subroutine shafftGetBuffers_sp(plan, alloc_size, data, work)
      type(c_ptr), intent(in) :: plan
      integer(c_size_t), intent(in) :: alloc_size
      complex(c_float), pointer, intent(out) :: data(:)
      complex(c_float), pointer, intent(out) :: work(:)
      type(c_ptr) :: cdata, cwork
      call shafftCheck(c_shafftGetBuffers(plan, cdata, cwork))
      call c_f_pointer(cdata, data, [alloc_size])
      call c_f_pointer(cwork, work, [alloc_size])
   end subroutine shafftGetBuffers_sp
   

   !> @brief Retrieve the currently attached buffers (double precision).
   !! @ingroup fortran_api
   !! @details
   !! **Important:** During execution the library may swap the roles of `data` and `work`.
   !! Call this **after** `shafftExecute` to locate the buffer that holds the transformed data.
   !! @param[in]  plan       Plan pointer.
   !! @param[in]  alloc_size Number of complex elements (from shafftGetAllocSize).
   !! @param[out] data       Complex(double) data array pointer (associated on return).
   !! @param[out] work       Complex(double) work array pointer (associated on return).
   subroutine shafftGetBuffers_dp(plan, alloc_size, data, work)
      type(c_ptr), intent(in) :: plan
      integer(c_size_t), intent(in) :: alloc_size
      complex(c_double), pointer, intent(out) :: data(:)
      complex(c_double), pointer, intent(out) :: work(:)
      type(c_ptr) :: cdata, cwork
      call shafftCheck(c_shafftGetBuffers(plan, cdata, cwork))
      call c_f_pointer(cdata, data, [alloc_size])
      call c_f_pointer(cwork, work, [alloc_size])
   end subroutine shafftGetBuffers_dp

   !> @brief Get the local subarray size and global offset for a specified layout.
   !! @ingroup fortran_api
   !! @details
   !! The layout may be:
   !! - `SHAFFT_TENSOR_LAYOUT_CURRENT`     : current layout (may be initial or transformed)
   !! - `SHAFFT_TENSOR_LAYOUT_INITIAL`     : initial layout (before any forward or backward transform)
   !! - `SHAFFT_TENSOR_LAYOUT_TRANSFORMED` : transformed layout (after forward or backward transform from initial)
   !! @param[in]  plan       Plan pointer.
   !! @param[out] subsize    Local extents per axis for this rank (length = ndim).
   !! @param[out] offset     Global starting indices per axis for this rank (length = ndim).
   !! @param[in]  layout     Layout specifier (see details).
   subroutine shafftGetLayout(plan, subsize, offset, layout)
      type(c_ptr), intent(in) :: plan
      integer(c_int), target, intent(out) :: subsize(:)
      integer(c_int), target, intent(out) :: offset(:)
      integer(c_int), intent(in) :: layout
      call shafftCheck(c_shafftGetLayout(plan, c_loc(subsize), c_loc(offset), layout))
   end subroutine shafftGetLayout

   !> @brief Get the locally contiguous and distributed axes for a specified layout.
   !! @ingroup fortran_api
   !! @details
   !! The layout may be:
   !! - `SHAFFT_TENSOR_LAYOUT_CURRENT`     : current layout (may be initial or transformed)
   !! - `SHAFFT_TENSOR_LAYOUT_INITIAL`     : initial layout (before any forward or backward transform)
   !! - `SHAFFT_TENSOR_LAYOUT_TRANSFORMED` : transformed layout (after forward or backward transform from initial)
   !! @param[in]  plan   Plan pointer.
   !! @param[out] ca    Indices of locally contiguous (non-distributed) axes, ordered innermost to outermost stride (length = ndim).
   !! @param[out] da    Indices of distributed axes (length = number of distributed axes).
   !! @param[in]  layout Layout specifier (see details).
   subroutine shafftGetAxes(plan, ca, da, layout)
      type(c_ptr), intent(in) :: plan
      integer(c_int), target, intent(out) :: ca(:)
      integer(c_int), target, intent(out) :: da(:)
      integer(c_int), intent(in) :: layout
      call shafftCheck(c_shafftGetAxes(plan, c_loc(ca), c_loc(da), layout))
   end subroutine shafftGetAxes

#if SHAFFT_BACKEND_HIPFFT
   !> @brief Set the HIP stream to use for all operations in the plan.
   !! @details
   !! The plan must have been created and must not have in-flight work.
   !! @param[in,out] plan   Plan pointer.
   !! @param[in]     stream HIP stream (type(c_ptr), e.g., obtained from hipStreamCreate).
   subroutine shafftSetStream(plan, stream)
      use iso_c_binding
      type(c_ptr), intent(inout) :: plan
      type(c_ptr), value :: stream
      call shafftCheck(c_shafftSetStream(plan, stream))
   end subroutine shafftSetStream
#endif

   !--------------------------------------------------------------------------
   ! Portable buffer allocation
   !--------------------------------------------------------------------------

   !> @brief Allocate a single-precision complex buffer suitable for the current backend.
   !! @ingroup fortran_api
   !! @param[in]  count Number of complex elements to allocate.
   !! @param[out] buf   Pointer that receives the allocated buffer.
   subroutine shafftAllocBuffer_sp(count, buf)
      use iso_c_binding
      integer(c_size_t), intent(in) :: count
      complex(c_float), pointer, intent(out) :: buf(:)
      type(c_ptr) :: cbuf
      call shafftCheck(c_shafftAllocBufferF(count, cbuf))
      call c_f_pointer(cbuf, buf, [count])
   end subroutine shafftAllocBuffer_sp

   !> @brief Allocate a double-precision complex buffer suitable for the current backend.
   !! @ingroup fortran_api
   !! @param[in]  count Number of complex elements to allocate.
   !! @param[out] buf   Pointer that receives the allocated buffer.
   subroutine shafftAllocBuffer_dp(count, buf)
      use iso_c_binding
      integer(c_size_t), intent(in) :: count
      complex(c_double), pointer, intent(out) :: buf(:)
      type(c_ptr) :: cbuf
      call shafftCheck(c_shafftAllocBufferD(count, cbuf))
      call c_f_pointer(cbuf, buf, [count])
   end subroutine shafftAllocBuffer_dp

   !> @brief Free a single-precision buffer allocated with shafftAllocBuffer.
   !! @ingroup fortran_api
   !! @param[in,out] buf Buffer to free; nullified on return.
   subroutine shafftFreeBuffer_sp(buf)
      use iso_c_binding
      complex(c_float), pointer, intent(inout) :: buf(:)
      if (associated(buf)) then
         call shafftCheck(c_shafftFreeBufferF(c_loc(buf)))
         nullify(buf)
      end if
   end subroutine shafftFreeBuffer_sp

   !> @brief Free a double-precision buffer allocated with shafftAllocBuffer.
   !! @ingroup fortran_api
   !! @param[in,out] buf Buffer to free; nullified on return.
   subroutine shafftFreeBuffer_dp(buf)
      use iso_c_binding
      complex(c_double), pointer, intent(inout) :: buf(:)
      if (associated(buf)) then
         call shafftCheck(c_shafftFreeBufferD(c_loc(buf)))
         nullify(buf)
      end if
   end subroutine shafftFreeBuffer_dp

   !--------------------------------------------------------------------------
   ! Portable memory copy
   !--------------------------------------------------------------------------

   !> @brief Copy single-precision data from host memory to a SHAFFT buffer.
   !! @ingroup fortran_api
   !! @param[out] dst   Destination buffer (allocated via shafftAllocBuffer).
   !! @param[in]  src   Source host array.
   !! @param[in]  count Number of complex elements to copy.
   subroutine shafftCopyToBuffer_sp(dst, src, count)
      use iso_c_binding
      complex(c_float), pointer, intent(inout) :: dst(:)
      complex(c_float), intent(in), target :: src(:)
      integer(c_size_t), intent(in) :: count
      call shafftCheck(c_shafftCopyToBufferF(c_loc(dst), c_loc(src), count))
   end subroutine shafftCopyToBuffer_sp

   !> @brief Copy double-precision data from host memory to a SHAFFT buffer.
   !! @ingroup fortran_api
   !! @param[out] dst   Destination buffer (allocated via shafftAllocBuffer).
   !! @param[in]  src   Source host array.
   !! @param[in]  count Number of complex elements to copy.
   subroutine shafftCopyToBuffer_dp(dst, src, count)
      use iso_c_binding
      complex(c_double), pointer, intent(inout) :: dst(:)
      complex(c_double), intent(in), target :: src(:)
      integer(c_size_t), intent(in) :: count
      call shafftCheck(c_shafftCopyToBufferD(c_loc(dst), c_loc(src), count))
   end subroutine shafftCopyToBuffer_dp

   !> @brief Copy single-precision data from a SHAFFT buffer to host memory.
   !! @ingroup fortran_api
   !! @param[out] dst   Destination host array.
   !! @param[in]  src   Source buffer.
   !! @param[in]  count Number of complex elements to copy.
   subroutine shafftCopyFromBuffer_sp(dst, src, count)
      use iso_c_binding
      complex(c_float), intent(out), target :: dst(:)
      complex(c_float), pointer, intent(in) :: src(:)
      integer(c_size_t), intent(in) :: count
      call shafftCheck(c_shafftCopyFromBufferF(c_loc(dst), c_loc(src), count))
   end subroutine shafftCopyFromBuffer_sp

   !> @brief Copy double-precision data from a SHAFFT buffer to host memory.
   !! @ingroup fortran_api
   !! @param[out] dst   Destination host array.
   !! @param[in]  src   Source buffer.
   !! @param[in]  count Number of complex elements to copy.
   subroutine shafftCopyFromBuffer_dp(dst, src, count)
      use iso_c_binding
      complex(c_double), intent(out), target :: dst(:)
      complex(c_double), pointer, intent(in) :: src(:)
      integer(c_size_t), intent(in) :: count
      call shafftCheck(c_shafftCopyFromBufferD(c_loc(dst), c_loc(src), count))
   end subroutine shafftCopyFromBuffer_dp

   !--------------------------------------------------------------------------
   ! Library information
   !--------------------------------------------------------------------------

   !> @brief Get the name of the FFT backend used at compile time.
   !! @ingroup fortran_api
   !! @param[out] name Character string receiving backend name ("FFTW" or "hipFFT").
   subroutine shafftGetBackendName(name)
      use iso_c_binding
      character(len=*), intent(out) :: name
      type(c_ptr) :: cname
      character(kind=c_char), pointer :: cstr(:)
      integer :: i, slen
      cname = c_shafftGetBackendName()
      slen = c_strlen(cname)
      call c_f_pointer(cname, cstr, [slen])
      name = ''
      do i = 1, min(slen, len(name))
         name(i:i) = cstr(i)
      end do
   end subroutine shafftGetBackendName

   !> @brief Get the library version as major, minor, patch components.
   !! @ingroup fortran_api
   !! @param[out] major Major version number.
   !! @param[out] minor Minor version number.
   !! @param[out] patch Patch version number.
   subroutine shafftGetVersion(major, minor, patch)
      use iso_c_binding
      integer(c_int), intent(out) :: major, minor, patch
      call c_shafftGetVersion(major, minor, patch)
   end subroutine shafftGetVersion

   !> @brief Get the library version as a string (e.g., "0.0.1-alpha").
   !! @ingroup fortran_api
   !! @param[out] version Character string receiving the version.
   subroutine shafftGetVersionString(version)
      use iso_c_binding
      character(len=*), intent(out) :: version
      type(c_ptr) :: cver
      character(kind=c_char), pointer :: cstr(:)
      integer :: i, slen
      cver = c_shafftGetVersionString()
      slen = c_strlen(cver)
      call c_f_pointer(cver, cstr, [slen])
      version = ''
      do i = 1, min(slen, len(version))
         version(i:i) = cstr(i)
      end do
   end subroutine shafftGetVersionString

   ! Helper function to get C string length
   function c_strlen(cptr) result(length)
      use iso_c_binding
      type(c_ptr), intent(in) :: cptr
      integer :: length
      character(kind=c_char), pointer :: cstr(:)
      integer :: i
      length = 0
      if (.not. c_associated(cptr)) return
      ! We'll limit to reasonable max length
      call c_f_pointer(cptr, cstr, [256])
      do i = 1, 256
         if (cstr(i) == c_null_char) exit
         length = length + 1
      end do
   end function c_strlen

   !> @brief Get the SHAFFT status code from the last error.
   function shafftLastErrorStatus() result(status)
      integer :: status
      status = c_shafft_last_error_status()
   end function shafftLastErrorStatus

   !> @brief Get the error source domain from the last error.
   function shafftLastErrorSource() result(source)
      integer :: source
      source = c_shafft_last_error_source()
   end function shafftLastErrorSource

   !> @brief Get the raw domain-specific error code from the last error.
   function shafftLastErrorDomainCode() result(code)
      integer :: code
      code = c_shafft_last_error_domain_code()
   end function shafftLastErrorDomainCode

   !> @brief Get a human-readable message for the last error.
   subroutine shafftLastErrorMessage(message)
      character(len=*), intent(out) :: message
      character(kind=c_char), target :: cbuf(256)
      integer(c_int) :: length
      integer :: i
      message = ''
      length = c_shafft_last_error_message(c_loc(cbuf), int(256, c_int))
      do i = 1, min(int(length), len(message))
         message(i:i) = cbuf(i)
      end do
   end subroutine shafftLastErrorMessage

   !> @brief Clear the last error state.
   subroutine shafftClearLastError()
      call c_shafft_clear_last_error()
   end subroutine shafftClearLastError

   !> @brief Get the name of an error source as a string.
   subroutine shafftErrorSourceName(source, name)
      integer, intent(in) :: source
      character(len=*), intent(out) :: name
      type(c_ptr) :: cname
      character(kind=c_char), pointer :: cstr(:)
      integer :: i, slen
      cname = c_shafft_error_source_name(int(source, c_int))
      slen = c_strlen(cname)
      name = ''
      if (slen > 0) then
         call c_f_pointer(cname, cstr, [slen])
         do i = 1, min(slen, len(name))
            name(i:i) = cstr(i)
         end do
      end if
   end subroutine shafftErrorSourceName

end module shafft
