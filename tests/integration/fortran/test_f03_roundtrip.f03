!> @file test_f03_roundtrip.f03
!! @brief Basic Fortran workflow test - forward/backward/normalize roundtrip
program test_f03_roundtrip
use iso_c_binding
use mpi
use shafft
use test_utils_f

implicit none

integer :: rank, nprocs, ierr
integer :: failed, passed

! Initialize MPI
call MPI_Init(ierr)
if (ierr /= MPI_SUCCESS) then
   write (*, '(A)') 'MPI_Init failed'
   call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
end if
call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

failed = 0
passed = 0

if (rank == 0) then
   write (*, '(A)') '=== Fortran Roundtrip Tests ==='
end if

! Run tests
call test_roundtrip_single_precision(passed, failed)
call test_roundtrip_double_precision(passed, failed)
call test_roundtrip_2d(passed, failed)

if (rank == 0) then
   write (*, '(A)') ''
   if (failed == 0) then
      write (*, '(A)') '=== Fortran Roundtrip Tests: PASSED ==='
   else
      write (*, '(A)') '=== Fortran Roundtrip Tests: FAILED ==='
   end if
   write (*, '(A,I0,A,I0)') 'Passed: ', passed, ', Failed: ', failed
end if

call MPI_Finalize(ierr)

if (failed > 0) then
   call exit(1)
end if

contains

subroutine test_roundtrip_single_precision(passed, failed)
   integer, intent(inout) :: passed, failed

   integer, parameter :: ndim = 3
   integer(c_int) :: dimensions(ndim), commDims(ndim)
   type(c_ptr) :: plan = c_null_ptr
   integer(c_size_t) :: localAllocSize, globalN
   complex(c_float), pointer :: data_buf(:) => null()
   complex(c_float), pointer :: work_buf(:) => null()
   complex(c_float), allocatable :: host_orig(:), host_result(:)
   integer :: i, rank, nprocs
   integer(c_int) :: e
   character(len=256) :: msg
   logical :: test_pass

   dimensions = [32, 32, 32]
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

   ! commDims for slab decomposition: [nprocs, 1, 1]
   commDims = [nprocs, 1, 1]

   if (rank == 0) write (*, '(A)', advance='no') '  roundtrip_single_precision               '

   ! Create and configure plan
   call shafftNDCreate(plan, e); if (e /= 0) goto 900
   call shafftNDInit(plan, commDims, dimensions, SHAFFT_C2C, MPI_COMM_WORLD, &
                     SHAFFT_LAYOUT_REDISTRIBUTED, e)
   if (e /= 0) goto 900
   call shafftGetAllocSize(plan, localAllocSize, e); if (e /= 0) goto 900

   ! Allocate buffers
   call shafftAllocBuffer(localAllocSize, data_buf, e); if (e /= 0) goto 900
   call shafftAllocBuffer(localAllocSize, work_buf, e); if (e /= 0) goto 900

   ! Initialize host data
   allocate (host_orig(localAllocSize))
   allocate (host_result(localAllocSize))
   do i = 1, int(localAllocSize)
      host_orig(i) = cmplx(real(mod(i - 1, 100), c_float)/100.0, &
                           real(mod(i + 49, 100), c_float)/100.0, kind=c_float)
   end do

   ! Copy to device and execute
   call shafftCopyToBuffer(data_buf, host_orig, localAllocSize, e); if (e /= 0) goto 900
   call shafftSetBuffers(plan, data_buf, work_buf, e); if (e /= 0) goto 900
   call shafftPlan(plan, e); if (e /= 0) goto 900
   call shafftExecute(plan, SHAFFT_FORWARD, e); if (e /= 0) goto 900
   call shafftExecute(plan, SHAFFT_BACKWARD, e); if (e /= 0) goto 900
   call shafftNormalize(plan, e); if (e /= 0) goto 900

   ! Get result
   call shafftGetBuffers(plan, localAllocSize, data_buf, work_buf, e); if (e /= 0) goto 900
   call shafftCopyFromBuffer(host_result, data_buf, localAllocSize, e); if (e /= 0) goto 900

   ! Compare using FFTW-style relative error with MPI sync
   globalN = product_int(dimensions)
   test_pass = checkRelErrorSP(host_result, host_orig, localAllocSize, globalN, &
                                   MPI_COMM_WORLD, real(TOL_F, c_double))
   test_pass = allRanksPassF(test_pass, MPI_COMM_WORLD)

   ! Cleanup
   deallocate (host_orig)
   deallocate (host_result)
   call shafftFreeBuffer(data_buf, e)
   call shafftFreeBuffer(work_buf, e)
   call shafftDestroy(plan, e)

   if (test_pass) then
      if (rank == 0) write (*, '(A)') 'PASS'
      passed = passed + 1
   else
      if (rank == 0) write (*, '(A)') 'FAIL (relative error exceeds tolerance)'
      failed = failed + 1
   end if
   return
900 continue
   failed = failed + 1
   if (rank == 0) then
      call shafftLastErrorMessage(msg)
      write (*, '(A,I0,3A)') 'FAIL (error code=', e, ', msg=', trim(msg), ')'
   end if
   if (associated(data_buf)) call shafftFreeBuffer(data_buf, e)
   if (associated(work_buf)) call shafftFreeBuffer(work_buf, e)
   if (c_associated(plan)) call shafftDestroy(plan, e)
end subroutine test_roundtrip_single_precision

subroutine test_roundtrip_double_precision(passed, failed)
   integer, intent(inout) :: passed, failed

   integer, parameter :: ndim = 2
   integer(c_int) :: dimensions(ndim), commDims(ndim)
   type(c_ptr) :: plan = c_null_ptr
   integer(c_size_t) :: localAllocSize, globalN
   complex(c_double), pointer :: data_buf(:) => null()
   complex(c_double), pointer :: work_buf(:) => null()
   complex(c_double), allocatable :: host_orig(:), host_result(:)
   integer :: i, rank, nprocs
   integer(c_int) :: e
   character(len=256) :: msg
   logical :: test_pass

   dimensions = [64, 64]
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

   ! commDims for slab decomposition: [nprocs, 1]
   commDims = [nprocs, 1]

   if (rank == 0) write (*, '(A)', advance='no') '  roundtrip_double_precision               '

   ! Create and configure plan
   call shafftNDCreate(plan, e); if (e /= 0) goto 901
   call shafftNDInit(plan, commDims, dimensions, SHAFFT_Z2Z, MPI_COMM_WORLD, &
                     SHAFFT_LAYOUT_REDISTRIBUTED, e)
   if (e /= 0) goto 901
   call shafftGetAllocSize(plan, localAllocSize, e); if (e /= 0) goto 901

   ! Allocate buffers
   call shafftAllocBuffer(localAllocSize, data_buf, e); if (e /= 0) goto 901
   call shafftAllocBuffer(localAllocSize, work_buf, e); if (e /= 0) goto 901

   ! Initialize host data
   allocate (host_orig(localAllocSize))
   allocate (host_result(localAllocSize))
   do i = 1, int(localAllocSize)
      host_orig(i) = cmplx(real(mod(i - 1, 100), c_double)/100.0d0, &
                           real(mod(i + 49, 100), c_double)/100.0d0, kind=c_double)
   end do

   ! Copy to device and execute
   call shafftCopyToBuffer(data_buf, host_orig, localAllocSize, e); if (e /= 0) goto 901
   call shafftSetBuffers(plan, data_buf, work_buf, e); if (e /= 0) goto 901
   call shafftPlan(plan, e); if (e /= 0) goto 901
   call shafftExecute(plan, SHAFFT_FORWARD, e); if (e /= 0) goto 901
   call shafftExecute(plan, SHAFFT_BACKWARD, e); if (e /= 0) goto 901
   call shafftNormalize(plan, e); if (e /= 0) goto 901

   ! Get result
   call shafftGetBuffers(plan, localAllocSize, data_buf, work_buf, e); if (e /= 0) goto 901
   call shafftCopyFromBuffer(host_result, data_buf, localAllocSize, e); if (e /= 0) goto 901

   ! Compare using FFTW-style relative error with MPI sync
   globalN = product_int(dimensions)
   test_pass = check_rel_error_dp(host_result, host_orig, localAllocSize, globalN, &
                                   MPI_COMM_WORLD, TOL_D)
   test_pass = allRanksPassF(test_pass, MPI_COMM_WORLD)

   ! Cleanup
   deallocate (host_orig)
   deallocate (host_result)
   call shafftFreeBuffer(data_buf, e)
   call shafftFreeBuffer(work_buf, e)
   call shafftDestroy(plan, e)

   if (test_pass) then
      if (rank == 0) write (*, '(A)') 'PASS'
      passed = passed + 1
   else
      if (rank == 0) write (*, '(A)') 'FAIL (relative error exceeds tolerance)'
      failed = failed + 1
   end if
   return
901 continue
   failed = failed + 1
   if (rank == 0) then
      call shafftLastErrorMessage(msg)
      write (*, '(A,I0,3A)') 'FAIL (error code=', e, ', msg=', trim(msg), ')'
   end if
   if (associated(data_buf)) call shafftFreeBuffer(data_buf, e)
   if (associated(work_buf)) call shafftFreeBuffer(work_buf, e)
   if (c_associated(plan)) call shafftDestroy(plan, e)
end subroutine test_roundtrip_double_precision

subroutine test_roundtrip_2d(passed, failed)
   integer, intent(inout) :: passed, failed

   integer, parameter :: ndim = 2
   integer(c_int) :: dimensions(ndim), commDims(ndim)
   type(c_ptr) :: plan = c_null_ptr
   integer(c_size_t) :: localAllocSize, globalN
   complex(c_float), pointer :: data_buf(:) => null()
   complex(c_float), pointer :: work_buf(:) => null()
   complex(c_float), allocatable :: host_orig(:), host_result(:)
   integer :: i, rank, nprocs
   integer(c_int) :: e
   character(len=256) :: msg
   logical :: test_pass

   dimensions = [128, 64]
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

   ! commDims for slab decomposition: [nprocs, 1]
   commDims = [nprocs, 1]

   if (rank == 0) write (*, '(A)', advance='no') '  roundtrip_2d                             '

   ! Create and configure plan
   call shafftNDCreate(plan, e); if (e /= 0) goto 902
   call shafftNDInit(plan, commDims, dimensions, SHAFFT_C2C, MPI_COMM_WORLD, &
                     SHAFFT_LAYOUT_REDISTRIBUTED, e)
   if (e /= 0) goto 902
   call shafftGetAllocSize(plan, localAllocSize, e); if (e /= 0) goto 902

   ! Allocate buffers
   call shafftAllocBuffer(localAllocSize, data_buf, e); if (e /= 0) goto 902
   call shafftAllocBuffer(localAllocSize, work_buf, e); if (e /= 0) goto 902

   ! Initialize host data
   allocate (host_orig(localAllocSize))
   allocate (host_result(localAllocSize))
   do i = 1, int(localAllocSize)
      host_orig(i) = cmplx(sin(real(i, c_float)*0.01), &
                           cos(real(i, c_float)*0.01), kind=c_float)
   end do

   ! Copy to device and execute
   call shafftCopyToBuffer(data_buf, host_orig, localAllocSize, e); if (e /= 0) goto 902
   call shafftSetBuffers(plan, data_buf, work_buf, e); if (e /= 0) goto 902
   call shafftPlan(plan, e); if (e /= 0) goto 902
   call shafftExecute(plan, SHAFFT_FORWARD, e); if (e /= 0) goto 902
   call shafftExecute(plan, SHAFFT_BACKWARD, e); if (e /= 0) goto 902
   call shafftNormalize(plan, e); if (e /= 0) goto 902

   ! Get result
   call shafftGetBuffers(plan, localAllocSize, data_buf, work_buf, e); if (e /= 0) goto 902
   call shafftCopyFromBuffer(host_result, data_buf, localAllocSize, e); if (e /= 0) goto 902

   ! Compare using FFTW-style relative error with MPI sync
   globalN = product_int(dimensions)
   test_pass = checkRelErrorSP(host_result, host_orig, localAllocSize, globalN, &
                                   MPI_COMM_WORLD, real(TOL_F, c_double))
   test_pass = allRanksPassF(test_pass, MPI_COMM_WORLD)

   ! Cleanup
   deallocate (host_orig)
   deallocate (host_result)
   call shafftFreeBuffer(data_buf, e)
   call shafftFreeBuffer(work_buf, e)
   call shafftDestroy(plan, e)

   if (test_pass) then
      if (rank == 0) write (*, '(A)') 'PASS'
      passed = passed + 1
   else
      if (rank == 0) write (*, '(A)') 'FAIL (relative error exceeds tolerance)'
      failed = failed + 1
   end if
   return
902 continue
   failed = failed + 1
   if (rank == 0) then
      call shafftLastErrorMessage(msg)
      write (*, '(A,I0,3A)') 'FAIL (error code=', e, ', msg=', trim(msg), ')'
   end if
   if (associated(data_buf)) call shafftFreeBuffer(data_buf, e)
   if (associated(work_buf)) call shafftFreeBuffer(work_buf, e)
   if (c_associated(plan)) call shafftDestroy(plan, e)
end subroutine test_roundtrip_2d

end program test_f03_roundtrip
