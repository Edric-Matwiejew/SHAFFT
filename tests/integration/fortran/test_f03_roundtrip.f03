!> @file test_f03_roundtrip.f03
!! @brief Basic Fortran workflow test - forward/backward/normalize roundtrip
program test_f03_roundtrip
use iso_c_binding
use mpi
use shafft

implicit none

integer :: rank, nprocs, ierr
integer :: failed, passed
real(c_float), parameter :: TOLERANCE = 1.0e-4

! Initialize MPI
call MPI_Init(ierr)
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
   integer, parameter :: nda = 1
   integer(c_int) :: dimensions(ndim)
   type(c_ptr) :: plan
   integer(c_size_t) :: alloc_size
   complex(c_float), pointer :: data_buf(:) => null()
   complex(c_float), pointer :: work_buf(:) => null()
   complex(c_float), allocatable :: host_orig(:), host_result(:)
   integer :: i, rank
   integer(c_int) :: e
   character(len=256) :: msg
   real(c_float) :: max_err, err_real, err_imag
   logical :: test_pass

   dimensions = [32, 32, 32]
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)

   if (rank == 0) write (*, '(A)', advance='no') '  roundtrip_single_precision               '

   ! Create plan
   call shafftPlanNDA(plan, ndim, nda, dimensions, SHAFFT_C2C, MPI_COMM_WORLD, e); if (e /= 0) goto 900
   call shafftGetAllocSize(plan, alloc_size, e); if (e /= 0) goto 900

   ! Allocate buffers
   call shafftAllocBuffer(alloc_size, data_buf, e); if (e /= 0) goto 900
   call shafftAllocBuffer(alloc_size, work_buf, e); if (e /= 0) goto 900

   ! Initialize host data
   allocate (host_orig(alloc_size))
   allocate (host_result(alloc_size))
   do i = 1, int(alloc_size)
      host_orig(i) = cmplx(real(mod(i - 1, 100), c_float)/100.0, &
                           real(mod(i + 49, 100), c_float)/100.0, kind=c_float)
   end do

   ! Copy to device and execute
   call shafftCopyToBuffer(data_buf, host_orig, alloc_size, e); if (e /= 0) goto 900
   call shafftSetBuffers(plan, data_buf, work_buf, e); if (e /= 0) goto 900
   call shafftExecute(plan, SHAFFT_FORWARD, e); if (e /= 0) goto 900
   call shafftExecute(plan, SHAFFT_BACKWARD, e); if (e /= 0) goto 900
   call shafftNormalize(plan, e); if (e /= 0) goto 900

   ! Get result
   call shafftGetBuffers(plan, alloc_size, data_buf, work_buf, e); if (e /= 0) goto 900
   call shafftCopyFromBuffer(host_result, data_buf, alloc_size, e); if (e /= 0) goto 900

   ! Compare
   max_err = 0.0
   do i = 1, int(alloc_size)
      err_real = abs(real(host_result(i)) - real(host_orig(i)))
      err_imag = abs(aimag(host_result(i)) - aimag(host_orig(i)))
      if (err_real > max_err) max_err = err_real
      if (err_imag > max_err) max_err = err_imag
   end do

   test_pass = (max_err < TOLERANCE)

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
      if (rank == 0) write (*, '(A,E10.3)') 'FAIL (max_err=', max_err, ')'
      failed = failed + 1
   end if
   return
900 continue
   failed = failed + 1
   if (rank == 0) then
      call shafftLastErrorMessage(msg)
      write (*, '(A,I0,A,A)') 'FAIL (error code=', e, ', msg=', trim(msg), ')'
   end if
end subroutine test_roundtrip_single_precision

subroutine test_roundtrip_double_precision(passed, failed)
   integer, intent(inout) :: passed, failed

   integer, parameter :: ndim = 2
   integer, parameter :: nda = 1
   integer(c_int) :: dimensions(ndim)
   type(c_ptr) :: plan
   integer(c_size_t) :: alloc_size
   complex(c_double), pointer :: data_buf(:) => null()
   complex(c_double), pointer :: work_buf(:) => null()
   complex(c_double), allocatable :: host_orig(:), host_result(:)
   integer :: i, rank
   integer(c_int) :: e
   character(len=256) :: msg
   real(c_double) :: max_err, err_real, err_imag
   real(c_double), parameter :: TOLERANCE_DP = 1.0d-10
   logical :: test_pass

   dimensions = [64, 64]
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)

   if (rank == 0) write (*, '(A)', advance='no') '  roundtrip_double_precision               '

   ! Create plan
   call shafftPlanNDA(plan, ndim, nda, dimensions, SHAFFT_Z2Z, MPI_COMM_WORLD, e); if (e /= 0) goto 901
   call shafftGetAllocSize(plan, alloc_size, e); if (e /= 0) goto 901

   ! Allocate buffers
   call shafftAllocBuffer(alloc_size, data_buf, e); if (e /= 0) goto 901
   call shafftAllocBuffer(alloc_size, work_buf, e); if (e /= 0) goto 901

   ! Initialize host data
   allocate (host_orig(alloc_size))
   allocate (host_result(alloc_size))
   do i = 1, int(alloc_size)
      host_orig(i) = cmplx(real(mod(i - 1, 100), c_double)/100.0d0, &
                           real(mod(i + 49, 100), c_double)/100.0d0, kind=c_double)
   end do

   ! Copy to device and execute
   call shafftCopyToBuffer(data_buf, host_orig, alloc_size, e); if (e /= 0) goto 901
   call shafftSetBuffers(plan, data_buf, work_buf, e); if (e /= 0) goto 901
   call shafftExecute(plan, SHAFFT_FORWARD, e); if (e /= 0) goto 901
   call shafftExecute(plan, SHAFFT_BACKWARD, e); if (e /= 0) goto 901
   call shafftNormalize(plan, e); if (e /= 0) goto 901

   ! Get result
   call shafftGetBuffers(plan, alloc_size, data_buf, work_buf, e); if (e /= 0) goto 901
   call shafftCopyFromBuffer(host_result, data_buf, alloc_size, e); if (e /= 0) goto 901

   ! Compare
   max_err = 0.0d0
   do i = 1, int(alloc_size)
      err_real = abs(real(host_result(i)) - real(host_orig(i)))
      err_imag = abs(aimag(host_result(i)) - aimag(host_orig(i)))
      if (err_real > max_err) max_err = err_real
      if (err_imag > max_err) max_err = err_imag
   end do

   test_pass = (max_err < TOLERANCE_DP)

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
      if (rank == 0) write (*, '(A,E10.3)') 'FAIL (max_err=', max_err, ')'
      failed = failed + 1
   end if
   return
901 continue
   failed = failed + 1
   if (rank == 0) then
      call shafftLastErrorMessage(msg)
      write (*, '(A,I0,A,A)') 'FAIL (error code=', e, ', msg=', trim(msg), ')'
   end if
end subroutine test_roundtrip_double_precision

subroutine test_roundtrip_2d(passed, failed)
   integer, intent(inout) :: passed, failed

   integer, parameter :: ndim = 2
   integer, parameter :: nda = 1
   integer(c_int) :: dimensions(ndim)
   type(c_ptr) :: plan
   integer(c_size_t) :: alloc_size
   complex(c_float), pointer :: data_buf(:) => null()
   complex(c_float), pointer :: work_buf(:) => null()
   complex(c_float), allocatable :: host_orig(:), host_result(:)
   integer :: i, rank
   integer(c_int) :: e
   character(len=256) :: msg
   real(c_float) :: max_err, err_real, err_imag
   logical :: test_pass

   dimensions = [128, 64]
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)

   if (rank == 0) write (*, '(A)', advance='no') '  roundtrip_2d                             '

   ! Create plan
   call shafftPlanNDA(plan, ndim, nda, dimensions, SHAFFT_C2C, MPI_COMM_WORLD, e); if (e /= 0) goto 902
   call shafftGetAllocSize(plan, alloc_size, e); if (e /= 0) goto 902

   ! Allocate buffers
   call shafftAllocBuffer(alloc_size, data_buf, e); if (e /= 0) goto 902
   call shafftAllocBuffer(alloc_size, work_buf, e); if (e /= 0) goto 902

   ! Initialize host data
   allocate (host_orig(alloc_size))
   allocate (host_result(alloc_size))
   do i = 1, int(alloc_size)
      host_orig(i) = cmplx(sin(real(i, c_float)*0.01), &
                           cos(real(i, c_float)*0.01), kind=c_float)
   end do

   ! Copy to device and execute
   call shafftCopyToBuffer(data_buf, host_orig, alloc_size, e); if (e /= 0) goto 902
   call shafftSetBuffers(plan, data_buf, work_buf, e); if (e /= 0) goto 902
   call shafftExecute(plan, SHAFFT_FORWARD, e); if (e /= 0) goto 902
   call shafftExecute(plan, SHAFFT_BACKWARD, e); if (e /= 0) goto 902
   call shafftNormalize(plan, e); if (e /= 0) goto 902

   ! Get result
   call shafftGetBuffers(plan, alloc_size, data_buf, work_buf, e); if (e /= 0) goto 902
   call shafftCopyFromBuffer(host_result, data_buf, alloc_size, e); if (e /= 0) goto 902

   ! Compare
   max_err = 0.0
   do i = 1, int(alloc_size)
      err_real = abs(real(host_result(i)) - real(host_orig(i)))
      err_imag = abs(aimag(host_result(i)) - aimag(host_orig(i)))
      if (err_real > max_err) max_err = err_real
      if (err_imag > max_err) max_err = err_imag
   end do

   test_pass = (max_err < TOLERANCE)

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
      if (rank == 0) write (*, '(A,E10.3)') 'FAIL (max_err=', max_err, ')'
      failed = failed + 1
   end if
   return
902 continue
   failed = failed + 1
   if (rank == 0) then
      call shafftLastErrorMessage(msg)
      write (*, '(A,I0,A,A)') 'FAIL (error code=', e, ', msg=', trim(msg), ')'
   end if
end subroutine test_roundtrip_2d

end program test_f03_roundtrip
