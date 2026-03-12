!> @file shafft.f03
!! @brief Fortran 2003 interface to SHAFFT.
!!
!! Axis indices are 0-based. Buffers may be swapped during execution;
!! call shafftGetBuffers after shafftExecute to locate the current result buffer.
module shafft
use iso_c_binding
#include "shafft_config.h"

implicit none

private

! Configuration subroutines
public :: shafftConfigurationND, shafftConfiguration1D

! Plan creation and destruction (N-D)
public :: shafftNDCreate, shafftNDInit
public :: shafftPlan
public :: shafftDestroy

! Plan creation and destruction (1D)
public :: shafft1DCreate, shafft1DInit

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

! Finalization
public :: shafftFinalize

! Config object lifecycle
public :: shafftConfigNDInit, shafftConfigNDRelease, shafftConfigNDResolve
public :: shafftNDInitFromConfig
public :: shafftConfig1DInit, shafftConfig1DRelease, shafftConfig1DResolve
public :: shafft1DInitFromConfig
public :: shafftGetCommunicator

! Config derived types
public :: shafft_config_nd_t, shafft_config_1d_t

!> @brief Fortran mirror of shafft_nd_config_t.
!! @ingroup fortran_api
type :: shafft_config_nd_t
  type(c_ptr) :: handle = c_null_ptr  ! opaque C config pointer

  ! ---- Read-only resolved fields (populated by init/resolve) ----
  integer :: ndim_ = 0
  integer :: precision_ = 0
  integer(c_size_t) :: allocElements = 0
  integer :: isActive = 0
  integer :: activeRank = -1
  integer :: activeSize = 0
  integer :: nda = 0
  integer :: commSize_ = 0
  integer :: status_ = 0
  integer :: flags = 0

  ! Communicators (MPI handles)
  integer :: worldComm = 0
  integer :: activeComm = 0

  ! Topology metadata
  integer :: nodeId = 0
  integer :: nodeCount = 0
  character(len=256) :: hostname = ' '
  integer(c_size_t) :: hostnameLen = 0
  character(len=256) :: deviceName = ' '
  integer(c_size_t) :: deviceNameLen = 0

  ! Arrays (allocated to ndim after init)
  integer(c_size_t), allocatable :: globalShape(:)
  integer, allocatable :: commDims(:)
  integer(c_size_t), allocatable :: initialSubsize(:)
  integer(c_size_t), allocatable :: initialOffset(:)
  integer(c_size_t), allocatable :: outputSubsize(:)
  integer(c_size_t), allocatable :: outputOffset(:)

  ! ---- Modifiable fields (edit, then call resolve) ----
  integer :: outputPolicy = 0
  integer :: strategy = 0
  integer(c_size_t) :: memLimit = 0
  integer :: hintNda = 0
  integer, allocatable :: hintCommDims(:)
end type shafft_config_nd_t

!> @brief Fortran mirror of shafft_1d_config_t.
!! @ingroup fortran_api
type :: shafft_config_1d_t
  type(c_ptr) :: handle = c_null_ptr  ! opaque C config pointer

  ! ---- Read-only resolved fields ----
  integer(c_size_t) :: globalSize = 0
  integer :: precision_ = 0
  integer(c_size_t) :: allocElements = 0
  integer :: isActive = 0
  integer :: activeRank = -1
  integer :: activeSize = 0
  integer :: status_ = 0
  integer :: flags = 0

  ! Communicators (MPI handles)
  integer :: worldComm = 0
  integer :: activeComm = 0

  ! Topology metadata
  integer :: nodeId = 0
  integer :: nodeCount = 0
  character(len=256) :: hostname = ' '
  integer(c_size_t) :: hostnameLen = 0
  character(len=256) :: deviceName = ' '
  integer(c_size_t) :: deviceNameLen = 0

  ! Layout fields
  integer(c_size_t) :: initialLocalSize  = 0
  integer(c_size_t) :: initialLocalStart = 0
  integer(c_size_t) :: outputLocalSize   = 0
  integer(c_size_t) :: outputLocalStart  = 0
end type shafft_config_1d_t

! Config flag constants
integer(c_int), parameter, public :: SHAFFT_CONFIG_CHANGED_COMM_DIMS = 1
integer(c_int), parameter, public :: SHAFFT_CONFIG_CHANGED_NDA = 2
integer(c_int), parameter, public :: SHAFFT_CONFIG_INACTIVE_RANKS = 4
integer(c_int), parameter, public :: SHAFFT_CONFIG_RESOLVED = 8

! FFT type constants
integer(c_int), parameter, public :: SHAFFT_C2C = 0 ! Single-precision complex-to-complex
integer(c_int), parameter, public :: SHAFFT_Z2Z = 1 ! Double-precision complex-to-complex

! Transform direction constants
integer(c_int), parameter, public :: SHAFFT_FORWARD = 0 ! Forward FFT
integer(c_int), parameter, public :: SHAFFT_BACKWARD = 1 ! Backward (inverse) FFT

! Tensor layout constants
integer(c_int), parameter, public :: SHAFFT_TENSOR_LAYOUT_CURRENT = 0 ! Current layout
integer(c_int), parameter, public :: SHAFFT_TENSOR_LAYOUT_INITIAL = 1 ! Initial layout
integer(c_int), parameter, public :: SHAFFT_TENSOR_LAYOUT_REDISTRIBUTED = 2 ! Redistributed

! Transform output policy constants
integer(c_int), parameter, public :: SHAFFT_LAYOUT_REDISTRIBUTED = 0 ! Keep post-forward layout
integer(c_int), parameter, public :: SHAFFT_LAYOUT_INITIAL = 1 ! Restore initial layout

! Error Source Constants
integer(c_int), parameter, public :: SHAFFT_ERRSRC_NONE = 0 ! No error or SHAFFT-internal error
integer(c_int), parameter, public :: SHAFFT_ERRSRC_MPI = 1 ! MPI library error
integer(c_int), parameter, public :: SHAFFT_ERRSRC_HIP = 2 ! HIP runtime error
integer(c_int), parameter, public :: SHAFFT_ERRSRC_HIPFFT = 3 ! hipFFT library error
integer(c_int), parameter, public :: SHAFFT_ERRSRC_FFTW = 4 ! FFTW library error
integer(c_int), parameter, public :: SHAFFT_ERRSRC_SYSTEM = 5 ! OS / allocation errors

! Status code constants (kept in sync with shafft_status_t)
integer(c_int), parameter, public :: SHAFFT_SUCCESS = 0
integer(c_int), parameter, public :: SHAFFT_ERR_NULLPTR = 1
integer(c_int), parameter, public :: SHAFFT_ERR_INVALID_COMM = 2
integer(c_int), parameter, public :: SHAFFT_ERR_NO_BUFFER = 3
integer(c_int), parameter, public :: SHAFFT_ERR_PLAN_NOT_INIT = 4
integer(c_int), parameter, public :: SHAFFT_ERR_INVALID_DIM = 5
integer(c_int), parameter, public :: SHAFFT_ERR_DIM_MISMATCH = 6
integer(c_int), parameter, public :: SHAFFT_ERR_INVALID_DECOMP = 7
integer(c_int), parameter, public :: SHAFFT_ERR_INVALID_FFTTYPE = 8
integer(c_int), parameter, public :: SHAFFT_ERR_ALLOC = 9
integer(c_int), parameter, public :: SHAFFT_ERR_BACKEND = 10
integer(c_int), parameter, public :: SHAFFT_ERR_MPI = 11
integer(c_int), parameter, public :: SHAFFT_ERR_INVALID_LAYOUT = 12
integer(c_int), parameter, public :: SHAFFT_ERR_SIZE_OVERFLOW = 13
integer(c_int), parameter, public :: SHAFFT_ERR_NOT_IMPL = 14
integer(c_int), parameter, public :: SHAFFT_ERR_INVALID_STATE = 15
integer(c_int), parameter, public :: SHAFFT_ERR_INTERNAL = 16

! Decomposition strategy constants
integer(c_int), parameter, public :: SHAFFT_MAXIMIZE_NDA = 0 ! Maximize distributed axes
integer(c_int), parameter, public :: SHAFFT_MINIMIZE_NDA = 1 ! Minimize distributed axes

interface

  ! ---- N-D Distributed FFT C bindings ---------------------------------

  function c_shafftConfigurationND(ndim, size, precision, commDims, nda, subsize, offset, &
                                   commSize, strategy, memLimit, comm) &
    bind(C, name="shafftConfigurationNDf03") result(status)
    use iso_c_binding
    integer(c_int), value :: ndim
    type(c_ptr), value :: size
    integer(kind(SHAFFT_C2C)), value :: precision
    type(c_ptr), value :: commDims
    integer(c_int) :: nda
    type(c_ptr), value :: subsize
    type(c_ptr), value :: offset
    integer(c_int) :: commSize
    integer(kind(SHAFFT_MAXIMIZE_NDA)), value :: strategy
    integer(c_size_t), value :: memLimit
    integer, intent(in) :: comm
    integer(c_int) :: status
  end function c_shafftConfigurationND

  function c_shafftNDCreate(outPlan) bind(C, name="shafftNDCreate") result(status)
    use iso_c_binding
    type(c_ptr) :: outPlan
    integer(c_int) :: status
  end function c_shafftNDCreate

  function c_shafftNDInit(plan, ndim, commDims, dimensions, precision, comm, outputPolicy) &
    bind(C, name="shafftNDInitf03") result(status)
    use iso_c_binding
    type(c_ptr), value :: plan
    integer(c_int), value :: ndim
    type(c_ptr), value :: commDims
    type(c_ptr), value :: dimensions
    integer(kind(SHAFFT_C2C)), value :: precision
    integer, intent(in) :: comm
    integer(c_int), intent(in) :: outputPolicy
    integer(c_int) :: status
  end function c_shafftNDInit

  ! ---- 1D Distributed FFT C bindings ----------------------------------

  function c_shafftConfiguration1D(N, localN, localStart, localAllocSize, precision, comm) &
    bind(C, name="shafftConfiguration1Df03") result(status)
    use iso_c_binding
    integer(c_size_t), value :: N
    integer(c_size_t) :: localN
    integer(c_size_t) :: localStart
    integer(c_size_t) :: localAllocSize
    integer(kind(SHAFFT_C2C)), value :: precision
    integer, intent(in) :: comm
    integer(c_int) :: status
  end function c_shafftConfiguration1D

  function c_shafft1DCreate(outPlan) bind(C, name="shafft1DCreate") result(status)
    use iso_c_binding
    type(c_ptr) :: outPlan
    integer(c_int) :: status
  end function c_shafft1DCreate

  function c_shafft1DInit(plan, N, localN, localStart, precision, comm) &
    bind(C, name="shafft1DInitf03") result(status)
    use iso_c_binding
    type(c_ptr), value :: plan
    integer(c_size_t), value :: N
    integer(c_size_t), value :: localN
    integer(c_size_t), value :: localStart
    integer(kind(SHAFFT_C2C)), value :: precision
    integer, intent(in) :: comm
    integer(c_int) :: status
  end function c_shafft1DInit

   function c_shafftGetAllocSize(plan, localAllocSize) &
    bind(C, name="shafftGetAllocSize") result(status)
    use iso_c_binding
    type(c_ptr), value :: plan
    integer(c_size_t) :: localAllocSize
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

  function c_shafftPlan(plan) bind(C, name="shafftPlan") result(status)
    use iso_c_binding
    type(c_ptr), value :: plan
    integer(c_int) :: status
  end function c_shafftPlan

      function c_shafftGetLayout(plan, subsize, offset, layout) &
    bind(C, name="shafftGetLayout") result(status)
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

      function c_shafftCopyFromBufferF(dst, src, count) &
    bind(C, name="shafftCopyFromBufferF") result(status)
    use iso_c_binding
    type(c_ptr), value :: dst
    type(c_ptr), value :: src
    integer(c_size_t), value :: count
    integer(c_int) :: status
  end function c_shafftCopyFromBufferF

      function c_shafftCopyFromBufferD(dst, src, count) &
    bind(C, name="shafftCopyFromBufferD") result(status)
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

  function c_shafftLastErrorStatus() bind(C, name="shafftLastErrorStatus") result(status)
    use iso_c_binding
    integer(c_int) :: status
  end function c_shafftLastErrorStatus

  function c_shafftLastErrorSource() bind(C, name="shafftLastErrorSource") result(source)
    use iso_c_binding
    integer(c_int) :: source
  end function c_shafftLastErrorSource

  function c_shafftLastErrorDomainCode() bind(C, name="shafftLastErrorDomainCode") result(code)
    use iso_c_binding
    integer(c_int) :: code
  end function c_shafftLastErrorDomainCode

function c_shafftLastErrorMessage(buf, buflen) bind(C, name="shafftLastErrorMessage") result(length)
    use iso_c_binding
    type(c_ptr), value :: buf
    integer(c_int), value :: buflen
    integer(c_int) :: length
  end function c_shafftLastErrorMessage

  subroutine c_shafftClearLastError() bind(C, name="shafftClearLastError")
    use iso_c_binding
  end subroutine c_shafftClearLastError

  function c_shafftErrorSourceName(source) bind(C, name="shafftErrorSourceName") result(name)
    use iso_c_binding
    integer(c_int), value :: source
    type(c_ptr) :: name
  end function c_shafftErrorSourceName

  ! ---- Finalization ---------------------------------------------------

  function c_shafftFinalize() bind(C, name="shafftFinalize") result(status)
    use iso_c_binding
    integer(c_int) :: status
  end function c_shafftFinalize

  ! ---- Config struct alloc/free helpers --------------------------------

  function c_shafftConfigNDAlloc() &
    bind(C, name="shafftConfigNDAllocf03") result(ptr)
    use iso_c_binding
    type(c_ptr) :: ptr
  end function c_shafftConfigNDAlloc

  subroutine c_shafftConfigNDFree(ptr) &
    bind(C, name="shafftConfigNDFreef03")
    use iso_c_binding
    type(c_ptr), value :: ptr
  end subroutine c_shafftConfigNDFree

  function c_shafftConfig1DAlloc() &
    bind(C, name="shafftConfig1DAllocf03") result(ptr)
    use iso_c_binding
    type(c_ptr) :: ptr
  end function c_shafftConfig1DAlloc

  subroutine c_shafftConfig1DFree(ptr) &
    bind(C, name="shafftConfig1DFreef03")
    use iso_c_binding
    type(c_ptr), value :: ptr
  end subroutine c_shafftConfig1DFree

  ! ---- Config object lifecycle C bindings -----------------------------

  function c_shafftConfigNDInit(cfg, ndim, globalShape, precision, commDims, &
                                 hintNda, strategy, outputPolicy, memLimit, comm) &
    bind(C, name="shafftConfigNDInitf03") result(status)
    use iso_c_binding
    type(c_ptr), value :: cfg
    integer(c_int), value :: ndim
    type(c_ptr), value :: globalShape
    integer(c_int), value :: precision
    type(c_ptr), value :: commDims
    integer(c_int), value :: hintNda
    integer(c_int), value :: strategy
    integer(c_int), value :: outputPolicy
    integer(c_size_t), value :: memLimit
    integer, intent(in) :: comm
    integer(c_int) :: status
  end function c_shafftConfigNDInit

  subroutine c_shafftConfigNDRelease(cfg) bind(C, name="shafftConfigNDRelease")
    use iso_c_binding
    type(c_ptr), value :: cfg
  end subroutine c_shafftConfigNDRelease

  function c_shafftConfigNDResolve(cfg) &
    bind(C, name="shafftConfigNDResolvef03") result(status)
    use iso_c_binding
    type(c_ptr), value :: cfg
    integer(c_int) :: status
  end function c_shafftConfigNDResolve

  function c_shafftNDInitFromConfig(plan, cfg) &
    bind(C, name="shafftNDInitFromConfigf03") result(status)
    use iso_c_binding
    type(c_ptr), value :: plan
    type(c_ptr), value :: cfg
    integer(c_int) :: status
  end function c_shafftNDInitFromConfig

  function c_shafftConfig1DInit(cfg, globalSize, precision, comm) &
    bind(C, name="shafftConfig1DInitf03") result(status)
    use iso_c_binding
    type(c_ptr), value :: cfg
    integer(c_size_t), value :: globalSize
    integer(c_int), value :: precision
    integer, intent(in) :: comm
    integer(c_int) :: status
  end function c_shafftConfig1DInit

  subroutine c_shafftConfig1DRelease(cfg) bind(C, name="shafftConfig1DRelease")
    use iso_c_binding
    type(c_ptr), value :: cfg
  end subroutine c_shafftConfig1DRelease

  function c_shafftConfig1DResolve(cfg) &
    bind(C, name="shafftConfig1DResolvef03") result(status)
    use iso_c_binding
    type(c_ptr), value :: cfg
    integer(c_int) :: status
  end function c_shafftConfig1DResolve

  function c_shafft1DInitFromConfig(plan, cfg) &
    bind(C, name="shafft1DInitFromConfigf03") result(status)
    use iso_c_binding
    type(c_ptr), value :: plan
    type(c_ptr), value :: cfg
    integer(c_int) :: status
  end function c_shafft1DInitFromConfig

  function c_shafftGetCommunicator(plan, outComm) &
    bind(C, name="shafftGetCommunicatorf03") result(status)
    use iso_c_binding
    type(c_ptr), value :: plan
    integer, intent(out) :: outComm
    integer(c_int) :: status
  end function c_shafftGetCommunicator

  ! ---- ND config accessor C bindings ----------------------------------

  ! ND scalar getters
  function c_shafftConfigNDGetNdim(cfg) bind(C, name="shafftConfigNDGetNdim") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_int) :: v
  end function
  function c_shafftConfigNDGetPrecision(cfg) bind(C, name="shafftConfigNDGetPrecision") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_int) :: v
  end function
  function c_shafftConfigNDGetAllocElements(cfg) &
    bind(C, name="shafftConfigNDGetAllocElements") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_size_t) :: v
  end function
  function c_shafftConfigNDGetIsActive(cfg) bind(C, name="shafftConfigNDGetIsActive") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_int) :: v
  end function
  function c_shafftConfigNDGetActiveRank(cfg) bind(C, name="shafftConfigNDGetActiveRank") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_int) :: v
  end function
  function c_shafftConfigNDGetActiveSize(cfg) bind(C, name="shafftConfigNDGetActiveSize") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_int) :: v
  end function
  function c_shafftConfigNDGetNda(cfg) bind(C, name="shafftConfigNDGetNda") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_int) :: v
  end function
  function c_shafftConfigNDGetCommSize(cfg) bind(C, name="shafftConfigNDGetCommSize") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_int) :: v
  end function
  function c_shafftConfigNDGetStatus(cfg) bind(C, name="shafftConfigNDGetStatus") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_int) :: v
  end function
  function c_shafftConfigNDGetOutputPolicy(cfg) &
    bind(C, name="shafftConfigNDGetOutputPolicy") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_int) :: v
  end function
  function c_shafftConfigNDGetStrategy(cfg) bind(C, name="shafftConfigNDGetStrategy") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_int) :: v
  end function
  function c_shafftConfigNDGetMemLimit(cfg) bind(C, name="shafftConfigNDGetMemLimit") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_size_t) :: v
  end function
  function c_shafftConfigNDGetHintNda(cfg) bind(C, name="shafftConfigNDGetHintNda") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_int) :: v
  end function
  function c_shafftConfigNDGetFlags(cfg) bind(C, name="shafftConfigNDGetFlags") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_int) :: v
  end function
  function c_shafftConfigNDGetWorldComm(cfg) bind(C, name="shafftConfigNDGetWorldComm") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_int) :: v
  end function
  function c_shafftConfigNDGetActiveComm(cfg) bind(C, name="shafftConfigNDGetActiveComm") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_int) :: v
  end function

  ! ND topology getters
  function c_shafftConfigNDGetNodeId(cfg) bind(C, name="shafftConfigNDGetNodeId") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_int) :: v
  end function
  function c_shafftConfigNDGetNodeCount(cfg) bind(C, name="shafftConfigNDGetNodeCount") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_int) :: v
  end function
  function c_shafftConfigNDGetHostnameLen(cfg) &
    bind(C, name="shafftConfigNDGetHostnameLen") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_size_t) :: v
  end function
  subroutine c_shafftConfigNDGetHostname(cfg, out, maxLen) &
    bind(C, name="shafftConfigNDGetHostname")
    use iso_c_binding
    type(c_ptr), value :: cfg
    character(c_char), intent(out) :: out(*)
    integer(c_size_t), value :: maxLen
  end subroutine
  function c_shafftConfigNDGetDeviceNameLen(cfg) &
    bind(C, name="shafftConfigNDGetDeviceNameLen") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_size_t) :: v
  end function
  subroutine c_shafftConfigNDGetDeviceName(cfg, out, maxLen) &
    bind(C, name="shafftConfigNDGetDeviceName")
    use iso_c_binding
    type(c_ptr), value :: cfg
    character(c_char), intent(out) :: out(*)
    integer(c_size_t), value :: maxLen
  end subroutine

  ! ND array getters
  subroutine c_shafftConfigNDGetGlobalShape(cfg, out) bind(C, name="shafftConfigNDGetGlobalShape")
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_size_t), intent(out) :: out(*)
  end subroutine
  subroutine c_shafftConfigNDGetCommDims(cfg, out) bind(C, name="shafftConfigNDGetCommDims")
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_int), intent(out) :: out(*)
  end subroutine
  subroutine c_shafftConfigNDGetHintCommDims(cfg, out) bind(C, name="shafftConfigNDGetHintCommDims")
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_int), intent(out) :: out(*)
  end subroutine
  subroutine c_shafftConfigNDGetInitialSubsize(cfg, out) &
    bind(C, name="shafftConfigNDGetInitialSubsize")
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_size_t), intent(out) :: out(*)
  end subroutine
  subroutine c_shafftConfigNDGetInitialOffset(cfg, out) &
    bind(C, name="shafftConfigNDGetInitialOffset")
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_size_t), intent(out) :: out(*)
  end subroutine
  subroutine c_shafftConfigNDGetOutputSubsize(cfg, out) &
    bind(C, name="shafftConfigNDGetOutputSubsize")
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_size_t), intent(out) :: out(*)
  end subroutine
  subroutine c_shafftConfigNDGetOutputOffset(cfg, out) bind(C, name="shafftConfigNDGetOutputOffset")
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_size_t), intent(out) :: out(*)
  end subroutine

  ! ND setters
  subroutine c_shafftConfigNDSetOutputPolicy(cfg, pol) bind(C, name="shafftConfigNDSetOutputPolicy")
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_int), value :: pol
  end subroutine
  subroutine c_shafftConfigNDSetStrategy(cfg, s) bind(C, name="shafftConfigNDSetStrategy")
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_int), value :: s
  end subroutine
  subroutine c_shafftConfigNDSetMemLimit(cfg, limit) bind(C, name="shafftConfigNDSetMemLimit")
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_size_t), value :: limit
  end subroutine
  subroutine c_shafftConfigNDSetHintNda(cfg, nda) bind(C, name="shafftConfigNDSetHintNda")
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_int), value :: nda
  end subroutine
  subroutine c_shafftConfigNDSetHintCommDims(cfg, dims) &
    bind(C, name="shafftConfigNDSetHintCommDims")
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_int), intent(in) :: dims(*)
  end subroutine

  ! ---- 1D config accessor C bindings ----------------------------------

  ! 1D scalar getters
  function c_shafftConfig1DGetGlobalSize(cfg) bind(C, name="shafftConfig1DGetGlobalSize") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_size_t) :: v
  end function
  function c_shafftConfig1DGetPrecision(cfg) bind(C, name="shafftConfig1DGetPrecision") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_int) :: v
  end function
  function c_shafftConfig1DGetAllocElements(cfg) &
    bind(C, name="shafftConfig1DGetAllocElements") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_size_t) :: v
  end function
  function c_shafftConfig1DGetIsActive(cfg) bind(C, name="shafftConfig1DGetIsActive") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_int) :: v
  end function
  function c_shafftConfig1DGetActiveRank(cfg) bind(C, name="shafftConfig1DGetActiveRank") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_int) :: v
  end function
  function c_shafftConfig1DGetActiveSize(cfg) bind(C, name="shafftConfig1DGetActiveSize") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_int) :: v
  end function
  function c_shafftConfig1DGetStatus(cfg) bind(C, name="shafftConfig1DGetStatus") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_int) :: v
  end function
  function c_shafftConfig1DGetFlags(cfg) bind(C, name="shafftConfig1DGetFlags") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_int) :: v
  end function
  function c_shafftConfig1DGetWorldComm(cfg) bind(C, name="shafftConfig1DGetWorldComm") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_int) :: v
  end function
  function c_shafftConfig1DGetActiveComm(cfg) bind(C, name="shafftConfig1DGetActiveComm") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_int) :: v
  end function

  ! 1D topology getters
  function c_shafftConfig1DGetNodeId(cfg) bind(C, name="shafftConfig1DGetNodeId") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_int) :: v
  end function
  function c_shafftConfig1DGetNodeCount(cfg) bind(C, name="shafftConfig1DGetNodeCount") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_int) :: v
  end function
  function c_shafftConfig1DGetHostnameLen(cfg) &
    bind(C, name="shafftConfig1DGetHostnameLen") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_size_t) :: v
  end function
  subroutine c_shafftConfig1DGetHostname(cfg, out, maxLen) &
    bind(C, name="shafftConfig1DGetHostname")
    use iso_c_binding
    type(c_ptr), value :: cfg
    character(c_char), intent(out) :: out(*)
    integer(c_size_t), value :: maxLen
  end subroutine
  function c_shafftConfig1DGetDeviceNameLen(cfg) &
    bind(C, name="shafftConfig1DGetDeviceNameLen") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_size_t) :: v
  end function
  subroutine c_shafftConfig1DGetDeviceName(cfg, out, maxLen) &
    bind(C, name="shafftConfig1DGetDeviceName")
    use iso_c_binding
    type(c_ptr), value :: cfg
    character(c_char), intent(out) :: out(*)
    integer(c_size_t), value :: maxLen
  end subroutine

  ! 1D layout getters
  function c_shafftConfig1DGetInitialLocalSize(cfg) &
    bind(C, name="shafftConfig1DGetInitialLocalSize") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_size_t) :: v
  end function
  function c_shafftConfig1DGetInitialLocalStart(cfg) &
    bind(C, name="shafftConfig1DGetInitialLocalStart") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_size_t) :: v
  end function
  function c_shafftConfig1DGetOutputLocalSize(cfg) &
    bind(C, name="shafftConfig1DGetOutputLocalSize") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_size_t) :: v
  end function
  function c_shafftConfig1DGetOutputLocalStart(cfg) &
    bind(C, name="shafftConfig1DGetOutputLocalStart") result(v)
    use iso_c_binding; type(c_ptr), value :: cfg; integer(c_size_t) :: v
  end function

end interface

!> @brief Generic interface to attach data/work buffers (single or double).
!! @ingroup fortran_api
!! Dispatches to precision-specific implementations.
interface shafftSetBuffers
  module procedure shafftSetBuffers_sp
  module procedure shafftSetBuffers_dp
end interface shafftSetBuffers

!> @brief Generic interface to retrieve current buffers (single or double).
!! @ingroup fortran_api
!! Buffers may be swapped after execute(); returns size inferred if not provided.
interface shafftGetBuffers
  module procedure shafftGetBuffers_sp
  module procedure shafftGetBuffers_dp
end interface shafftGetBuffers

!> @brief Generic interface to allocate a SHAFFT buffer (single or double).
!! @ingroup fortran_api
interface shafftAllocBuffer
  module procedure shafftAllocBuffer_sp
  module procedure shafftAllocBuffer_dp
end interface shafftAllocBuffer

!> @brief Generic interface to free a SHAFFT buffer (single or double).
!! @ingroup fortran_api
interface shafftFreeBuffer
  module procedure shafftFreeBuffer_sp
  module procedure shafftFreeBuffer_dp
end interface shafftFreeBuffer

!> @brief Generic interface to copy host data into a SHAFFT buffer.
!! @ingroup fortran_api
interface shafftCopyToBuffer
  module procedure shafftCopyToBuffer_sp
  module procedure shafftCopyToBuffer_dp
end interface shafftCopyToBuffer

!> @brief Generic interface to copy from a SHAFFT buffer back to host data.
!! @ingroup fortran_api
interface shafftCopyFromBuffer
  module procedure shafftCopyFromBuffer_sp
  module procedure shafftCopyFromBuffer_dp
end interface shafftCopyFromBuffer

contains

!> @brief Compute local layout and process grid for N-D FFT.
   !! @ingroup fortran_api
   !!
   !! Preference: commDims > nda > strategy.
   !!
   !! @param[in]     ndim           Number of dimensions.
   !! @param[in]     globalShape    Global extents (length = ndim).
   !! @param[in]     precision      SHAFFT_C2C or SHAFFT_Z2Z.
   !! @param[in,out] commDims       Process grid (zeros for auto).
   !! @param[in,out] nda            Distributed axes (0 for auto).
   !! @param[out]    subsize        Local extents for this rank.
   !! @param[out]    offset         Global offsets for this rank.
   !! @param[out]    commSize       Active rank count.
   !! @param[in]     strategy       SHAFFT_MAXIMIZE_NDA or SHAFFT_MINIMIZE_NDA.
   !! @param[in]     memLimit       Memory limit in bytes (0 = none).
   !! @param[in]     comm           MPI communicator.
subroutine shafftConfigurationND(globalShape, precision, commDims, nda, subsize, offset, &
                                 commSize, strategy, memLimit, comm, ierr)
  integer(c_int), dimension(:), target, intent(in) :: globalShape
  integer(kind(SHAFFT_C2C)), intent(in) :: precision
  integer(c_int), dimension(:), target, intent(inout) :: commDims
  integer(c_int), intent(inout) :: nda
  integer(c_size_t), dimension(:), target, intent(out) :: subsize
  integer(c_size_t), dimension(:), target, intent(out) :: offset
  integer(c_int), intent(out) :: commSize
  integer(kind(SHAFFT_MAXIMIZE_NDA)), intent(in) :: strategy
  integer(c_size_t), intent(in) :: memLimit
  integer(c_int), intent(in) :: comm
  integer(c_int), intent(out) :: ierr
  integer(c_int) :: ndim

  ndim = size(globalShape, 1)

  ierr = c_shafftConfigurationND(ndim, c_loc(globalShape), precision, c_loc(commDims), nda, &
                                 c_loc(subsize), c_loc(offset), commSize, strategy, &
                                 memLimit, comm)
end subroutine shafftConfigurationND

!> @brief Compute local layout for 1D distributed FFT.
   !! @ingroup fortran_api
   !!
   !! @param[in]  N              Global FFT size.
   !! @param[out] localN         Elements for this rank.
   !! @param[out] localStart     This rank's offset.
   !! @param[out] localAllocSize Required buffer size.
   !! @param[in]  precision      SHAFFT_C2C or SHAFFT_Z2Z.
   !! @param[in]  comm           MPI communicator.
subroutine shafftConfiguration1D(N, localN, localStart, localAllocSize, precision, comm, ierr)
  integer(c_size_t), intent(in) :: N
  integer(c_size_t), intent(out) :: localN
  integer(c_size_t), intent(out) :: localStart
  integer(c_size_t), intent(out) :: localAllocSize
  integer(kind(SHAFFT_C2C)), intent(in) :: precision
  integer(c_int), intent(in) :: comm
  integer(c_int), intent(out) :: ierr

  ierr = c_shafftConfiguration1D(N, localN, localStart, localAllocSize, precision, comm)
end subroutine shafftConfiguration1D

!> @brief Allocate uninitialized N-D plan.
   !! @ingroup fortran_api
   !! @param[out] plan Plan pointer.
subroutine shafftNDCreate(plan, ierr)
  type(c_ptr), intent(out) :: plan
  integer(c_int), intent(out) :: ierr
  ierr = c_shafftNDCreate(plan)
end subroutine shafftNDCreate

!> @brief Initialise N-D plan (legacy direct-init path).
   !! @ingroup fortran_api
   !! @param[in,out] plan          Plan pointer.
   !! @param[in]     commDims      Processor grid.
   !! @param[in]     dimensions    Global FFT dimensions.
   !! @param[in]     precision     SHAFFT_C2C or SHAFFT_Z2Z.
   !! @param[in]     comm          MPI communicator.
   !! @param[in]     outputPolicy  Forward output-layout policy.
   !! @param[out]    ierr          Error code (0 on success).
subroutine shafftNDInit(plan, commDims, dimensions, precision, comm, outputPolicy, ierr)
  type(c_ptr), intent(inout) :: plan
  integer(c_int), target, intent(in) :: commDims(:)
  integer(c_int), target, intent(in) :: dimensions(:)
  integer(kind(SHAFFT_C2C)), intent(in) :: precision
  integer(c_int), intent(in) :: comm
  integer(c_int), intent(in) :: outputPolicy
  integer(c_int), intent(out) :: ierr
  integer(c_int) :: ndim

  ndim = size(dimensions, 1)

  ierr = c_shafftNDInit(plan, ndim, c_loc(commDims), c_loc(dimensions), &
                        precision, comm, outputPolicy)

end subroutine shafftNDInit

!> @brief Allocate uninitialized 1D plan.
   !! @ingroup fortran_api
   !! @param[out] plan Plan pointer.
subroutine shafft1DCreate(plan, ierr)
  type(c_ptr), intent(out) :: plan
  integer(c_int), intent(out) :: ierr
  ierr = c_shafft1DCreate(plan)
end subroutine shafft1DCreate

!> @brief Initialise 1D plan (legacy direct-init path).
   !! @ingroup fortran_api
   !! @param[in,out] plan       Plan pointer.
   !! @param[in]     N          Global FFT size.
   !! @param[in]     localN     Elements for this rank.
   !! @param[in]     localStart This rank's offset.
   !! @param[in]     precision  SHAFFT_C2C or SHAFFT_Z2Z.
   !! @param[in]     comm       MPI communicator.
   !! @param[out]    ierr       Error code (0 on success).
subroutine shafft1DInit(plan, N, localN, localStart, precision, comm, ierr)
  type(c_ptr), intent(inout) :: plan
  integer(c_size_t), intent(in) :: N
  integer(c_size_t), intent(in) :: localN
  integer(c_size_t), intent(in) :: localStart
  integer(kind(SHAFFT_C2C)), intent(in) :: precision
  integer(c_int), intent(in) :: comm
  integer(c_int), intent(out) :: ierr

  ierr = c_shafft1DInit(plan, N, localN, localStart, precision, comm)
end subroutine shafft1DInit

!> @brief Execute FFT transform.
   !! @ingroup fortran_api
   !!
   !! @param[in,out] plan      Plan pointer.
   !! @param[in]     direction SHAFFT_FORWARD or SHAFFT_BACKWARD.
   !! @param[out]    ierr      Error code (0 on success).
subroutine shafftExecute(plan, direction, ierr)
  type(c_ptr), intent(inout) :: plan
  integer(c_int), intent(in) :: direction
  integer(c_int), intent(out) :: ierr
  ierr = c_shafftExecute(plan, direction)
end subroutine shafftExecute

!> @brief Release plan resources.
   !! @ingroup fortran_api
   !!
   !! Sets plan to c_null_ptr on return.
   !! @param[in,out] plan Plan pointer.
   !! @param[out]    ierr Error code (0 on success).
subroutine shafftDestroy(plan, ierr)
  use iso_c_binding
  type(c_ptr), intent(inout) :: plan
  integer(c_int), intent(out) :: ierr
  ierr = c_shafftDestroy(plan)
end subroutine shafftDestroy

!> @brief Create backend FFT plans.
   !! @ingroup fortran_api
   !!
   !! Must be called after init (shafftNDInit or shafft1DInit).
   !! For N-D plans, output policy is configured in shafftNDInit().
   !! Works on N-D and 1D plans. Calling plan() more than once is an error.
   !! @param[in]  plan  Plan pointer.
   !! @param[out] ierr  Error code (0 on success).
subroutine shafftPlan(plan, ierr)
  use iso_c_binding
  type(c_ptr), value :: plan
  integer(c_int), intent(out) :: ierr
  ierr = c_shafftPlan(plan)
end subroutine shafftPlan

!> @brief Apply normalization to data buffer.
   !! @ingroup fortran_api
   !!
   !! @param[in,out] plan Plan pointer.
subroutine shafftNormalize(plan, ierr)
  type(c_ptr), intent(inout) :: plan
  integer(c_int), intent(out) :: ierr
  ierr = c_shafftNormalize(plan)
end subroutine shafftNormalize

!> @brief Get required buffer size in complex elements.
   !! @ingroup fortran_api
   !!
   !! @param[in]  plan           Plan pointer.
   !! @param[out] localAllocSize Element count.
subroutine shafftGetAllocSize(plan, localAllocSize, ierr)
  type(c_ptr), intent(in) :: plan
  integer(c_size_t), intent(out) :: localAllocSize
  integer(c_int), intent(out) :: ierr
  ierr = c_shafftGetAllocSize(plan, localAllocSize)
end subroutine shafftGetAllocSize

!> @brief Attach data and work buffers (single precision).
   !! @ingroup fortran_api
   !!
   !! @param[in,out] plan  Plan pointer.
   !! @param[in,out] fdata Data buffer.
   !! @param[in,out] fwork Work buffer.
subroutine shafftSetBuffers_sp(plan, fdata, fwork, ierr)
  type(c_ptr), intent(inout) :: plan
  complex(c_float), pointer, intent(inout) :: fdata(:)
  complex(c_float), pointer, intent(inout) :: fwork(:)
  integer(c_int), intent(out) :: ierr
  ierr = c_shafftSetBuffers(plan, c_loc(fdata), c_loc(fwork))
end subroutine shafftSetBuffers_sp

!> @brief Attach data and work buffers (double precision).
   !! @ingroup fortran_api
   !!
   !! @param[in,out] plan  Plan pointer.
   !! @param[in,out] fdata Data buffer.
   !! @param[in,out] fwork Work buffer.
subroutine shafftSetBuffers_dp(plan, fdata, fwork, ierr)
  type(c_ptr), intent(inout) :: plan
  complex(c_double), pointer, intent(inout) :: fdata(:)
  complex(c_double), pointer, intent(inout) :: fwork(:)
  integer(c_int), intent(out) :: ierr
  ierr = c_shafftSetBuffers(plan, c_loc(fdata), c_loc(fwork))
end subroutine shafftSetBuffers_dp

!> @brief Retrieve current buffer pointers (single precision).
   !! @ingroup fortran_api
   !!
   !! Buffers may be swapped after execute().
   !!
   !! @param[in]  plan           Plan pointer.
   !! @param[in]  localAllocSize Element count (optional).
   !! @param[out] data           Data buffer.
   !! @param[out] work           Work buffer.
subroutine shafftGetBuffers_sp(plan, localAllocSize, data, work, ierr)
  type(c_ptr), intent(in) :: plan
  integer(c_size_t), intent(in), optional :: localAllocSize
  complex(c_float), pointer, intent(out) :: data(:)
  complex(c_float), pointer, intent(out) :: work(:)
  type(c_ptr) :: cdata, cwork
  integer(c_size_t) :: n
  integer(c_int), intent(out) :: ierr

  ierr = c_shafftGetBuffers(plan, cdata, cwork)
  if (present(localAllocSize)) then
    n = localAllocSize
  else
    ierr = c_shafftGetAllocSize(plan, n)
    if (ierr /= 0) return
  end if
  call c_f_pointer(cdata, data, [n])
  call c_f_pointer(cwork, work, [n])
end subroutine shafftGetBuffers_sp

!> @brief Retrieve current buffer pointers (double precision).
   !! @ingroup fortran_api
   !!
   !! Buffers may be swapped after execute().
   !!
   !! @param[in]  plan           Plan pointer.
   !! @param[in]  localAllocSize Element count (optional).
   !! @param[out] data           Data buffer.
   !! @param[out] work           Work buffer.
subroutine shafftGetBuffers_dp(plan, localAllocSize, data, work, ierr)
  type(c_ptr), intent(in) :: plan
  integer(c_size_t), intent(in), optional :: localAllocSize
  complex(c_double), pointer, intent(out) :: data(:)
  complex(c_double), pointer, intent(out) :: work(:)
  type(c_ptr) :: cdata, cwork
  integer(c_size_t) :: n
  integer(c_int), intent(out) :: ierr

  ierr = c_shafftGetBuffers(plan, cdata, cwork)
  if (present(localAllocSize)) then
    n = localAllocSize
  else
    ierr = c_shafftGetAllocSize(plan, n)
    if (ierr /= 0) return
  end if
  call c_f_pointer(cdata, data, [n])
  call c_f_pointer(cwork, work, [n])
end subroutine shafftGetBuffers_dp

!> @brief Query local tensor layout.
   !! @ingroup fortran_api
   !!
   !! @param[in]  plan    Plan pointer.
   !! @param[out] subsize Local extents.
   !! @param[out] offset  Global offsets.
   !! @param[in]  layout  SHAFFT_TENSOR_LAYOUT_*.
subroutine shafftGetLayout(plan, subsize, offset, layout, ierr)
  type(c_ptr), intent(in) :: plan
  integer(c_size_t), target, intent(out) :: subsize(:)
  integer(c_size_t), target, intent(out) :: offset(:)
  integer(c_int), intent(in) :: layout
  integer(c_int), intent(out) :: ierr
  ierr = c_shafftGetLayout(plan, c_loc(subsize), c_loc(offset), layout)
end subroutine shafftGetLayout

!> @brief Query contiguous and distributed axis indices.
   !! @ingroup fortran_api
   !!
   !! @param[in]  plan   Plan pointer.
   !! @param[out] ca     Contiguous axis indices.
   !! @param[out] da     Distributed axis indices.
   !! @param[in]  layout SHAFFT_TENSOR_LAYOUT_*.
subroutine shafftGetAxes(plan, ca, da, layout, ierr)
  type(c_ptr), intent(in) :: plan
  integer(c_int), target, intent(out) :: ca(:)
  integer(c_int), target, intent(out) :: da(:)
  integer(c_int), intent(in) :: layout
  integer(c_int), intent(out) :: ierr
  ierr = c_shafftGetAxes(plan, c_loc(ca), c_loc(da), layout)
end subroutine shafftGetAxes

#if SHAFFT_BACKEND_HIPFFT
!> @brief Set HIP stream for GPU operations.
   !! @ingroup fortran_api
   !!
   !! @param[in,out] plan   Plan pointer.
   !! @param[in]     stream HIP stream.
subroutine shafftSetStream(plan, stream, ierr)
  use iso_c_binding
  type(c_ptr), intent(inout) :: plan
  type(c_ptr), value :: stream
  integer(c_int), intent(out) :: ierr
  ierr = c_shafftSetStream(plan, stream)
end subroutine shafftSetStream
#endif

!--------------------------------------------------------------------------
! Portable buffer allocation
!--------------------------------------------------------------------------

!> @brief Allocate single-precision complex buffer.
   !! @ingroup fortran_api
   !!
   !! @param[in]  count Number of complex elements.
   !! @param[out] buf   Allocated buffer.
subroutine shafftAllocBuffer_sp(count, buf, ierr)
  use iso_c_binding
  integer(c_size_t), intent(in) :: count
  complex(c_float), pointer, intent(out) :: buf(:)
  type(c_ptr) :: cbuf
  integer(c_int), intent(out) :: ierr
  ierr = c_shafftAllocBufferF(count, cbuf)
  call c_f_pointer(cbuf, buf, [count])
end subroutine shafftAllocBuffer_sp

!> @brief Allocate double-precision complex buffer.
   !! @ingroup fortran_api
   !!
   !! @param[in]  count Number of complex elements.
   !! @param[out] buf   Allocated buffer.
subroutine shafftAllocBuffer_dp(count, buf, ierr)
  use iso_c_binding
  integer(c_size_t), intent(in) :: count
  complex(c_double), pointer, intent(out) :: buf(:)
  type(c_ptr) :: cbuf
  integer(c_int), intent(out) :: ierr
  ierr = c_shafftAllocBufferD(count, cbuf)
  call c_f_pointer(cbuf, buf, [count])
end subroutine shafftAllocBuffer_dp

!> @brief Free single-precision buffer.
   !! @ingroup fortran_api
   !!
   !! @param[in,out] buf Buffer to free (nullified on return).
subroutine shafftFreeBuffer_sp(buf, ierr)
  use iso_c_binding
  complex(c_float), pointer, intent(inout) :: buf(:)
  integer(c_int), intent(out) :: ierr
  if (associated(buf)) then
    ierr = c_shafftFreeBufferF(c_loc(buf))
    nullify (buf)
  else
    ierr = 0
  end if
end subroutine shafftFreeBuffer_sp

!> @brief Free double-precision buffer.
   !! @ingroup fortran_api
   !!
   !! @param[in,out] buf Buffer to free (nullified on return).
subroutine shafftFreeBuffer_dp(buf, ierr)
  use iso_c_binding
  complex(c_double), pointer, intent(inout) :: buf(:)
  integer(c_int), intent(out) :: ierr
  if (associated(buf)) then
    ierr = c_shafftFreeBufferD(c_loc(buf))
    nullify (buf)
  else
    ierr = 0
  end if
end subroutine shafftFreeBuffer_dp

!--------------------------------------------------------------------------
! Portable memory copy
!--------------------------------------------------------------------------

!> @brief Copy single-precision data from host to buffer.
   !! @ingroup fortran_api
   !!
   !! @param[out] dst   Destination buffer.
   !! @param[in]  src   Source host array.
   !! @param[in]  count Element count (optional).
subroutine shafftCopyToBuffer_sp(dst, src, count, ierr)
  use iso_c_binding
  complex(c_float), pointer, intent(inout) :: dst(:)
  complex(c_float), intent(in), target :: src(:)
  integer(c_size_t), intent(in), optional :: count
  integer(c_size_t) :: n
  integer(c_int), intent(out) :: ierr
  if (present(count)) then
    n = count
  else
    n = int(size(src), c_size_t)
  end if
  ierr = c_shafftCopyToBufferF(c_loc(dst), c_loc(src), n)
end subroutine shafftCopyToBuffer_sp

!> @brief Copy double-precision data from host to buffer.
   !! @ingroup fortran_api
   !!
   !! @param[out] dst   Destination buffer.
   !! @param[in]  src   Source host array.
   !! @param[in]  count Element count (optional).
subroutine shafftCopyToBuffer_dp(dst, src, count, ierr)
  use iso_c_binding
  complex(c_double), pointer, intent(inout) :: dst(:)
  complex(c_double), intent(in), target :: src(:)
  integer(c_size_t), intent(in), optional :: count
  integer(c_size_t) :: n
  integer(c_int), intent(out) :: ierr
  if (present(count)) then
    n = count
  else
    n = int(size(src), c_size_t)
  end if
  ierr = c_shafftCopyToBufferD(c_loc(dst), c_loc(src), n)
end subroutine shafftCopyToBuffer_dp

!> @brief Copy single-precision data from buffer to host.
   !! @ingroup fortran_api
   !!
   !! @param[out] dst   Destination host array.
   !! @param[in]  src   Source buffer.
   !! @param[in]  count Element count (optional).
subroutine shafftCopyFromBuffer_sp(dst, src, count, ierr)
  use iso_c_binding
  complex(c_float), intent(out), target :: dst(:)
  complex(c_float), pointer, intent(in) :: src(:)
  integer(c_size_t), intent(in), optional :: count
  integer(c_size_t) :: n
  integer(c_int), intent(out) :: ierr
  if (present(count)) then
    n = count
  else
    n = int(size(dst), c_size_t)
  end if
  ierr = c_shafftCopyFromBufferF(c_loc(dst), c_loc(src), n)
end subroutine shafftCopyFromBuffer_sp

!> @brief Copy double-precision data from buffer to host.
   !! @ingroup fortran_api
   !!
   !! @param[out] dst   Destination host array.
   !! @param[in]  src   Source buffer.
   !! @param[in]  count Element count (optional).
subroutine shafftCopyFromBuffer_dp(dst, src, count, ierr)
  use iso_c_binding
  complex(c_double), intent(out), target :: dst(:)
  complex(c_double), pointer, intent(in) :: src(:)
  integer(c_size_t), intent(in), optional :: count
  integer(c_size_t) :: n
  integer(c_int), intent(out) :: ierr
  if (present(count)) then
    n = count
  else
    n = int(size(dst), c_size_t)
  end if
  ierr = c_shafftCopyFromBufferD(c_loc(dst), c_loc(src), n)
end subroutine shafftCopyFromBuffer_dp

!--------------------------------------------------------------------------
! Library information
!--------------------------------------------------------------------------

!> @brief Get FFT backend name.
   !! @ingroup fortran_api
   !!
   !! @param[out] name "FFTW" or "hipFFT".
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

!> @brief Get library version components.
   !! @ingroup fortran_api
   !!
   !! @param[out] major Major version.
   !! @param[out] minor Minor version.
   !! @param[out] patch Patch version.
subroutine shafftGetVersion(major, minor, patch)
  use iso_c_binding
  integer(c_int), intent(out) :: major, minor, patch
  call c_shafftGetVersion(major, minor, patch)
end subroutine shafftGetVersion

!> @brief Get library version string.
   !! @ingroup fortran_api
   !!
   !! @param[out] version Version string (e.g., "1.1.0-alpha").
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

!> @brief Get status code from last error.
   !! @ingroup fortran_api
   !!
   !! @return SHAFFT status code (0 = success).
function shafftLastErrorStatus() result(status)
  integer(c_int) :: status
  status = c_shafftLastErrorStatus()
end function shafftLastErrorStatus

!> @brief Get error source domain from last error.
   !! @ingroup fortran_api
   !!
   !! @return Error source code (SHAFFT_ERRSRC_*).
function shafftLastErrorSource() result(source)
  integer(c_int) :: source
  source = c_shafftLastErrorSource()
end function shafftLastErrorSource

!> @brief Get raw domain-specific error code from last error.
   !! @ingroup fortran_api
   !!
   !! @return Domain-specific error code (MPI/hipFFT/HIP/etc.).
function shafftLastErrorDomainCode() result(code)
  integer(c_int) :: code
  code = c_shafftLastErrorDomainCode()
end function shafftLastErrorDomainCode

!> @brief Get human-readable message for last error.
   !! @ingroup fortran_api
   !!
   !! @param[out] message Buffer to receive the message (truncated to length).
subroutine shafftLastErrorMessage(message)
  character(len=*), intent(out) :: message
  character(kind=c_char), target :: cbuf(256)
  integer(c_int) :: length
  integer :: i
  message = ''
  length = c_shafftLastErrorMessage(c_loc(cbuf), int(256, c_int))
  do i = 1, min(int(length), len(message))
    message(i:i) = cbuf(i)
  end do
end subroutine shafftLastErrorMessage

!> @brief Clear last error state.
   !! @ingroup fortran_api
   !!
   !! Clears the thread-local error record; does not modify message buffers.
subroutine shafftClearLastError()
  call c_shafftClearLastError()
end subroutine shafftClearLastError

!> @brief Get error source name as string.
   !! @ingroup fortran_api
   !!
   !! @param[in]  source Error source code (SHAFFT_ERRSRC_*).
   !! @param[out] name   Name of the source (e.g., "MPI", "HIP").
subroutine shafftErrorSourceName(source, name)
  integer(c_int), intent(in) :: source
  character(len=*), intent(out) :: name
  type(c_ptr) :: cname
  character(kind=c_char), pointer :: cstr(:)
  integer :: i, slen
  cname = c_shafftErrorSourceName(source)
  slen = c_strlen(cname)
  name = ''
  if (slen > 0) then
    call c_f_pointer(cname, cstr, [slen])
    do i = 1, min(slen, len(name))
      name(i:i) = cstr(i)
    end do
  end if
end subroutine shafftErrorSourceName

!> @brief Finalize library and release backend resources.
   !! @ingroup fortran_api
   !!
   !! Call after all plans are destroyed. Must be called before MPI_Finalize()
   !! for FFTW backend. Safe to call multiple times.
   !!
   !! @param[out] ierr Status code (0 on success).
subroutine shafftFinalize(ierr)
  integer(c_int), intent(out) :: ierr
  ierr = c_shafftFinalize()
end subroutine shafftFinalize

! ---- Config object lifecycle wrappers ------------------------------------

!> @brief Initialize an N-D configuration object.
   !! @ingroup fortran_api
   !!
   !! Performs init + resolve in one call. Config owns communicators.
   !!
   !! @param[in,out] cfg          Config derived type.
   !! @param[in]     ndim         Number of dimensions (>= 1).
   !! @param[in]     globalShape  Global extents (length = ndim).
   !! @param[in]     precision    SHAFFT_C2C or SHAFFT_Z2Z.
   !! @param[in]     commDims     Process grid (zeros for auto, or C_NULL_PTR).
   !! @param[in]     hintNda      Hint for number of distributed axes (0 = auto).
   !! @param[in]     strategy     SHAFFT_MAXIMIZE_NDA or SHAFFT_MINIMIZE_NDA.
   !! @param[in]     outputPolicy SHAFFT_LAYOUT_REDISTRIBUTED or SHAFFT_LAYOUT_INITIAL.
   !! @param[in]     memLimit     Memory limit in bytes (0 = none).
   !! @param[in]     comm         MPI communicator.
   !! @param[out]    ierr         Error code (0 on success).
subroutine shafftConfigNDInit(cfg, ndim, globalShape, precision, commDims, &
                              hintNda, strategy, outputPolicy, memLimit, comm, ierr)
  type(shafft_config_nd_t), intent(inout) :: cfg
  integer(c_int), intent(in) :: ndim
  integer(c_size_t), intent(in), target :: globalShape(ndim)
  integer(c_int), intent(in) :: precision
  integer(c_int), intent(in), target, optional :: commDims(ndim)
  integer(c_int), intent(in) :: hintNda
  integer(c_int), intent(in) :: strategy
  integer(c_int), intent(in) :: outputPolicy
  integer(c_size_t), intent(in) :: memLimit
  integer, intent(in) :: comm
  integer(c_int), intent(out) :: ierr
  type(c_ptr) :: pCommDims

  ! Allocate the C-side config struct on the heap
  cfg%handle = c_shafftConfigNDAlloc()
  if (.not. c_associated(cfg%handle)) then
    ierr = 1  ! SHAFFT_ERR_NULLPTR
    return
  end if

  if (present(commDims)) then
    pCommDims = c_loc(commDims)
  else
    pCommDims = c_null_ptr
  end if

  ierr = c_shafftConfigNDInit(cfg%handle, ndim, c_loc(globalShape), precision, &
                               pCommDims, hintNda, strategy, outputPolicy, &
                               memLimit, comm)
  if (ierr == SHAFFT_SUCCESS) then
    call config_nd_sync_from_c(cfg)
  else
    ! Clean up on failure
    call c_shafftConfigNDFree(cfg%handle)
    cfg%handle = c_null_ptr
  end if
end subroutine shafftConfigNDInit

!> @brief Release N-D config resources.
   !! @ingroup fortran_api
   !! @param[in,out] cfg Config derived type.
subroutine shafftConfigNDRelease(cfg)
  type(shafft_config_nd_t), intent(inout) :: cfg
  if (c_associated(cfg%handle)) then
    call c_shafftConfigNDRelease(cfg%handle)
    call c_shafftConfigNDFree(cfg%handle)
  end if
  if (allocated(cfg%globalShape))    deallocate(cfg%globalShape)
  if (allocated(cfg%commDims))       deallocate(cfg%commDims)
  if (allocated(cfg%hintCommDims))   deallocate(cfg%hintCommDims)
  if (allocated(cfg%initialSubsize)) deallocate(cfg%initialSubsize)
  if (allocated(cfg%initialOffset))  deallocate(cfg%initialOffset)
  if (allocated(cfg%outputSubsize))  deallocate(cfg%outputSubsize)
  if (allocated(cfg%outputOffset))   deallocate(cfg%outputOffset)
  cfg%handle = c_null_ptr
end subroutine shafftConfigNDRelease

!> @brief Re-resolve N-D configuration (collective).
   !! @ingroup fortran_api
   !!
   !! Pushes modifiable fields to C, re-resolves, then syncs back.
   !! Uses the worldComm stored in the config.
   !!
   !! @param[in,out] cfg  Initialized config derived type.
   !! @param[out]    ierr Error code (0 on success).
subroutine shafftConfigNDResolve(cfg, ierr)
  type(shafft_config_nd_t), intent(inout) :: cfg
  integer(c_int), intent(out) :: ierr
  call config_nd_sync_to_c(cfg)
  ierr = c_shafftConfigNDResolve(cfg%handle)
  if (ierr == SHAFFT_SUCCESS) call config_nd_sync_from_c(cfg)
end subroutine shafftConfigNDResolve

!> @brief Initialize N-D plan from resolved config.
   !! @ingroup fortran_api
   !!
   !! @param[in,out] plan Plan from shafftNDCreate.
   !! @param[in,out] cfg  Resolved config derived type.
   !! @param[out]    ierr Error code (0 on success).
subroutine shafftNDInitFromConfig(plan, cfg, ierr)
  type(c_ptr), intent(inout) :: plan
  type(shafft_config_nd_t), intent(inout) :: cfg
  integer(c_int), intent(out) :: ierr
  ierr = c_shafftNDInitFromConfig(plan, cfg%handle)
end subroutine shafftNDInitFromConfig

!> @brief Initialize a 1-D configuration object.
   !! @ingroup fortran_api
   !!
   !! Performs init + resolve in one call. Config owns communicators.
   !!
   !! @param[in,out] cfg        Config derived type.
   !! @param[in]     globalSize Global FFT length.
   !! @param[in]     precision  SHAFFT_C2C or SHAFFT_Z2Z.
   !! @param[in]     comm       MPI communicator.
   !! @param[out]    ierr       Error code (0 on success).
subroutine shafftConfig1DInit(cfg, globalSize, precision, comm, ierr)
  type(shafft_config_1d_t), intent(inout) :: cfg
  integer(c_size_t), intent(in) :: globalSize
  integer(c_int), intent(in) :: precision
  integer, intent(in) :: comm
  integer(c_int), intent(out) :: ierr

  ! Allocate the C-side config struct on the heap
  cfg%handle = c_shafftConfig1DAlloc()
  if (.not. c_associated(cfg%handle)) then
    ierr = 1  ! SHAFFT_ERR_NULLPTR
    return
  end if

  ierr = c_shafftConfig1DInit(cfg%handle, globalSize, precision, comm)
  if (ierr == SHAFFT_SUCCESS) then
    call config_1d_sync_from_c(cfg)
  else
    call c_shafftConfig1DFree(cfg%handle)
    cfg%handle = c_null_ptr
  end if
end subroutine shafftConfig1DInit

!> @brief Release 1-D config resources.
   !! @ingroup fortran_api
   !! @param[in,out] cfg Config derived type.
subroutine shafftConfig1DRelease(cfg)
  type(shafft_config_1d_t), intent(inout) :: cfg
  if (c_associated(cfg%handle)) then
    call c_shafftConfig1DRelease(cfg%handle)
    call c_shafftConfig1DFree(cfg%handle)
  end if
  cfg%handle = c_null_ptr
end subroutine shafftConfig1DRelease

!> @brief Re-resolve 1-D configuration (collective).
   !! @ingroup fortran_api
   !!
   !! Uses the worldComm stored in the config.
   !!
   !! @param[in,out] cfg  Initialized config derived type.
   !! @param[out]    ierr Error code (0 on success).
subroutine shafftConfig1DResolve(cfg, ierr)
  type(shafft_config_1d_t), intent(inout) :: cfg
  integer(c_int), intent(out) :: ierr
  ierr = c_shafftConfig1DResolve(cfg%handle)
  if (ierr == SHAFFT_SUCCESS) call config_1d_sync_from_c(cfg)
end subroutine shafftConfig1DResolve

!> @brief Initialize 1-D plan from resolved config.
   !! @ingroup fortran_api
   !!
   !! @param[in,out] plan Plan from shafft1DCreate.
   !! @param[in,out] cfg  Resolved config derived type.
   !! @param[out]    ierr Error code (0 on success).
subroutine shafft1DInitFromConfig(plan, cfg, ierr)
  type(c_ptr), intent(inout) :: plan
  type(shafft_config_1d_t), intent(inout) :: cfg
  integer(c_int), intent(out) :: ierr
  ierr = c_shafft1DInitFromConfig(plan, cfg%handle)
end subroutine shafft1DInitFromConfig

!> @brief Get duplicated communicator from plan.
   !! @ingroup fortran_api
   !!
   !! Returns MPI_COMM_NULL for inactive ranks. Caller must free
   !! the returned communicator with MPI_Comm_free.
   !!
   !! @param[in]  plan    Plan pointer (N-D or 1D).
   !! @param[out] outComm MPI communicator (Fortran integer handle).
   !! @param[out] ierr    Error code (0 on success).
subroutine shafftGetCommunicator(plan, outComm, ierr)
  type(c_ptr), intent(in) :: plan
  integer(c_int), intent(out) :: outComm
  integer(c_int), intent(out) :: ierr
  ierr = c_shafftGetCommunicator(plan, outComm)
end subroutine shafftGetCommunicator

! ---- Internal sync subroutines (module-private) --------------------------

!> @brief Pull resolved data from C struct into Fortran derived type (ND).
subroutine config_nd_sync_from_c(cfg)
  type(shafft_config_nd_t), intent(inout) :: cfg
  integer :: n

  ! Scalar fields
  cfg%ndim_         = c_shafftConfigNDGetNdim(cfg%handle)
  cfg%precision_    = c_shafftConfigNDGetPrecision(cfg%handle)
  cfg%allocElements = c_shafftConfigNDGetAllocElements(cfg%handle)
  cfg%isActive      = c_shafftConfigNDGetIsActive(cfg%handle)
  cfg%activeRank    = c_shafftConfigNDGetActiveRank(cfg%handle)
  cfg%activeSize    = c_shafftConfigNDGetActiveSize(cfg%handle)
  cfg%nda           = c_shafftConfigNDGetNda(cfg%handle)
  cfg%commSize_     = c_shafftConfigNDGetCommSize(cfg%handle)
  cfg%status_       = c_shafftConfigNDGetStatus(cfg%handle)
  cfg%outputPolicy  = c_shafftConfigNDGetOutputPolicy(cfg%handle)
  cfg%strategy      = c_shafftConfigNDGetStrategy(cfg%handle)
  cfg%memLimit      = c_shafftConfigNDGetMemLimit(cfg%handle)
  cfg%hintNda       = c_shafftConfigNDGetHintNda(cfg%handle)
  cfg%flags         = c_shafftConfigNDGetFlags(cfg%handle)
  cfg%worldComm     = c_shafftConfigNDGetWorldComm(cfg%handle)
  cfg%activeComm    = c_shafftConfigNDGetActiveComm(cfg%handle)

  ! Topology metadata
  cfg%nodeId        = c_shafftConfigNDGetNodeId(cfg%handle)
  cfg%nodeCount     = c_shafftConfigNDGetNodeCount(cfg%handle)
  cfg%hostnameLen   = c_shafftConfigNDGetHostnameLen(cfg%handle)
  call c_shafftConfigNDGetHostname(cfg%handle, cfg%hostname, int(256, c_size_t))
  cfg%deviceNameLen = c_shafftConfigNDGetDeviceNameLen(cfg%handle)
  call c_shafftConfigNDGetDeviceName(cfg%handle, cfg%deviceName, int(256, c_size_t))

  ! Array fields (allocate to ndim)
  n = cfg%ndim_
  if (n > 0) then
    if (.not. allocated(cfg%globalShape))    allocate(cfg%globalShape(n))
    if (.not. allocated(cfg%commDims))       allocate(cfg%commDims(n))
    if (.not. allocated(cfg%hintCommDims))   allocate(cfg%hintCommDims(n))
    if (.not. allocated(cfg%initialSubsize)) allocate(cfg%initialSubsize(n))
    if (.not. allocated(cfg%initialOffset))  allocate(cfg%initialOffset(n))
    if (.not. allocated(cfg%outputSubsize))  allocate(cfg%outputSubsize(n))
    if (.not. allocated(cfg%outputOffset))   allocate(cfg%outputOffset(n))

    call c_shafftConfigNDGetGlobalShape(cfg%handle, cfg%globalShape)
    call c_shafftConfigNDGetCommDims(cfg%handle, cfg%commDims)
    call c_shafftConfigNDGetHintCommDims(cfg%handle, cfg%hintCommDims)
    call c_shafftConfigNDGetInitialSubsize(cfg%handle, cfg%initialSubsize)
    call c_shafftConfigNDGetInitialOffset(cfg%handle, cfg%initialOffset)
    call c_shafftConfigNDGetOutputSubsize(cfg%handle, cfg%outputSubsize)
    call c_shafftConfigNDGetOutputOffset(cfg%handle, cfg%outputOffset)
  end if
end subroutine config_nd_sync_from_c

!> @brief Push user-modifiable fields from Fortran to C struct (ND).
subroutine config_nd_sync_to_c(cfg)
  type(shafft_config_nd_t), intent(inout) :: cfg
  call c_shafftConfigNDSetOutputPolicy(cfg%handle, cfg%outputPolicy)
  call c_shafftConfigNDSetStrategy(cfg%handle, cfg%strategy)
  call c_shafftConfigNDSetMemLimit(cfg%handle, cfg%memLimit)
  call c_shafftConfigNDSetHintNda(cfg%handle, cfg%hintNda)
  if (allocated(cfg%hintCommDims)) then
    call c_shafftConfigNDSetHintCommDims(cfg%handle, cfg%hintCommDims)
  end if
end subroutine config_nd_sync_to_c

!> @brief Pull resolved data from C struct into Fortran derived type (1D).
subroutine config_1d_sync_from_c(cfg)
  type(shafft_config_1d_t), intent(inout) :: cfg

  cfg%globalSize    = c_shafftConfig1DGetGlobalSize(cfg%handle)
  cfg%precision_    = c_shafftConfig1DGetPrecision(cfg%handle)
  cfg%allocElements = c_shafftConfig1DGetAllocElements(cfg%handle)
  cfg%isActive      = c_shafftConfig1DGetIsActive(cfg%handle)
  cfg%activeRank    = c_shafftConfig1DGetActiveRank(cfg%handle)
  cfg%activeSize    = c_shafftConfig1DGetActiveSize(cfg%handle)
  cfg%status_       = c_shafftConfig1DGetStatus(cfg%handle)
  cfg%flags         = c_shafftConfig1DGetFlags(cfg%handle)
  cfg%worldComm     = c_shafftConfig1DGetWorldComm(cfg%handle)
  cfg%activeComm    = c_shafftConfig1DGetActiveComm(cfg%handle)

  ! Topology metadata
  cfg%nodeId        = c_shafftConfig1DGetNodeId(cfg%handle)
  cfg%nodeCount     = c_shafftConfig1DGetNodeCount(cfg%handle)
  cfg%hostnameLen   = c_shafftConfig1DGetHostnameLen(cfg%handle)
  call c_shafftConfig1DGetHostname(cfg%handle, cfg%hostname, int(256, c_size_t))
  cfg%deviceNameLen = c_shafftConfig1DGetDeviceNameLen(cfg%handle)
  call c_shafftConfig1DGetDeviceName(cfg%handle, cfg%deviceName, int(256, c_size_t))

  ! Layout fields
  cfg%initialLocalSize  = c_shafftConfig1DGetInitialLocalSize(cfg%handle)
  cfg%initialLocalStart = c_shafftConfig1DGetInitialLocalStart(cfg%handle)
  cfg%outputLocalSize   = c_shafftConfig1DGetOutputLocalSize(cfg%handle)
  cfg%outputLocalStart  = c_shafftConfig1DGetOutputLocalStart(cfg%handle)
end subroutine config_1d_sync_from_c

!> @brief No-op — 1D has no user-modifiable fields.
subroutine config_1d_sync_to_c(cfg)
  type(shafft_config_1d_t), intent(inout) :: cfg
  ! nothing to push
end subroutine config_1d_sync_to_c

end module shafft
