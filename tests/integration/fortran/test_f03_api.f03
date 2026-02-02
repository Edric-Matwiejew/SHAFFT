!> @file test_f03_api.f03
!! @brief Comprehensive Fortran API coverage test
program test_f03_api
use iso_c_binding
use mpi
use shafft

implicit none

integer :: rank, nprocs, ierr
integer(c_int) :: e
integer :: failed, passed

! Initialize MPI
call MPI_Init(ierr)
call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

failed = 0
passed = 0

if (rank == 0) then
   write (*, '(A)') '=== Fortran API Coverage Tests ==='
end if

! Library info tests
call test_get_backend_name(passed, failed)
call test_get_version(passed, failed)
call test_get_version_string(passed, failed)

! Error API tests
call test_last_error_status(passed, failed)
call test_error_source_name(passed, failed)
call test_clear_last_error(passed, failed)

! Plan lifecycle tests
call test_plan_nda(passed, failed)
call test_plan_cart(passed, failed)
call test_plan_destroy(passed, failed)

! Buffer management tests
call test_alloc_free_buffer_sp(passed, failed)
call test_alloc_free_buffer_dp(passed, failed)
call test_copy_buffers_sp(passed, failed)
call test_copy_buffers_dp(passed, failed)

! Query tests
call test_get_alloc_size(passed, failed)
call test_get_layout(passed, failed)
call test_get_axes(passed, failed)
call test_set_get_buffers(passed, failed)

! Configuration tests
call test_configuration_nda(passed, failed)
call test_configuration_cart(passed, failed)

if (rank == 0) then
   write (*, '(A)') ''
   if (failed == 0) then
      write (*, '(A)') '=== Fortran API Coverage Tests: PASSED ==='
   else
      write (*, '(A)') '=== Fortran API Coverage Tests: FAILED ==='
   end if
   write (*, '(A,I0,A,I0)') 'Passed: ', passed, ', Failed: ', failed
end if

call MPI_Finalize(ierr)

if (failed > 0) then
   call exit(1)
end if

contains

!============================================================================
! Library Info Tests
!============================================================================

subroutine test_get_backend_name(passed, failed)
   integer, intent(inout) :: passed, failed
   character(len=32) :: backend_name
   integer :: rank

   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   if (rank == 0) write (*, '(A)', advance='no') '  shafftGetBackendName                     '

   call shafftGetBackendName(backend_name)

   if (len_trim(backend_name) > 0) then
      if (rank == 0) write (*, '(A)') 'PASS'
      passed = passed + 1
   else
      if (rank == 0) write (*, '(A)') 'FAIL (empty name)'
      failed = failed + 1
   end if
end subroutine test_get_backend_name

subroutine test_get_version(passed, failed)
   integer, intent(inout) :: passed, failed
   integer :: major, minor, patch, rank

   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   if (rank == 0) write (*, '(A)', advance='no') '  shafftGetVersion                         '

   call shafftGetVersion(major, minor, patch)

   if (major >= 0 .and. minor >= 0 .and. patch >= 0) then
      if (rank == 0) write (*, '(A)') 'PASS'
      passed = passed + 1
   else
      if (rank == 0) write (*, '(A)') 'FAIL (negative version)'
      failed = failed + 1
   end if
end subroutine test_get_version

subroutine test_get_version_string(passed, failed)
   integer, intent(inout) :: passed, failed
   character(len=32) :: version_str
   integer :: rank

   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   if (rank == 0) write (*, '(A)', advance='no') '  shafftGetVersionString                   '

   call shafftGetVersionString(version_str)

   if (len_trim(version_str) > 0 .and. index(version_str, '.') > 0) then
      if (rank == 0) write (*, '(A)') 'PASS'
      passed = passed + 1
   else
      if (rank == 0) write (*, '(A)') 'FAIL (invalid version string)'
      failed = failed + 1
   end if
end subroutine test_get_version_string

!============================================================================
! Error API Tests
!============================================================================

subroutine test_last_error_status(passed, failed)
   integer, intent(inout) :: passed, failed
   integer :: status, rank

   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   if (rank == 0) write (*, '(A)', advance='no') '  shafftLastErrorStatus                    '

   call shafftClearLastError()
   status = shafftLastErrorStatus()

   if (status == 0) then ! SHAFFT_SUCCESS
      if (rank == 0) write (*, '(A)') 'PASS'
      passed = passed + 1
   else
      if (rank == 0) write (*, '(A)') 'FAIL (expected SUCCESS)'
      failed = failed + 1
   end if
end subroutine test_last_error_status

subroutine test_error_source_name(passed, failed)
   integer, intent(inout) :: passed, failed
   character(len=32) :: name
   integer :: rank

   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   if (rank == 0) write (*, '(A)', advance='no') '  shafftErrorSourceName                    '

   call shafftErrorSourceName(SHAFFT_ERRSRC_NONE, name)

   if (len_trim(name) > 0) then
      if (rank == 0) write (*, '(A)') 'PASS'
      passed = passed + 1
   else
      if (rank == 0) write (*, '(A)') 'FAIL (empty name)'
      failed = failed + 1
   end if
end subroutine test_error_source_name

subroutine test_clear_last_error(passed, failed)
   integer, intent(inout) :: passed, failed
   integer :: status, rank

   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   if (rank == 0) write (*, '(A)', advance='no') '  shafftClearLastError                     '

   call shafftClearLastError()
   status = shafftLastErrorStatus()

   if (status == 0) then
      if (rank == 0) write (*, '(A)') 'PASS'
      passed = passed + 1
   else
      if (rank == 0) write (*, '(A)') 'FAIL (not cleared)'
      failed = failed + 1
   end if
end subroutine test_clear_last_error

!============================================================================
! Plan Lifecycle Tests
!============================================================================

subroutine test_plan_nda(passed, failed)
   integer, intent(inout) :: passed, failed
   integer, parameter :: ndim = 3, nda = 1
   integer(c_int) :: dimensions(ndim)
   type(c_ptr) :: plan
   integer :: rank
   integer(c_int) :: rc
   character(len=256) :: msg

   dimensions = [32, 32, 32]
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   if (rank == 0) write (*, '(A)', advance='no') '  shafftPlanNDA                            '

   call shafftPlanNDA(plan, ndim, nda, dimensions, SHAFFT_C2C, MPI_COMM_WORLD, rc)

   if (rc == 0 .and. c_associated(plan)) then
      if (rank == 0) write (*, '(A)') 'PASS'
      passed = passed + 1
      call shafftDestroy(plan, rc)
   else
      if (rank == 0) then
         call shafftLastErrorMessage(msg)
         write (*, '(A,I0,A,A)') 'FAIL (rc=', rc, ', msg=', trim(msg), ')'
      end if
      failed = failed + 1
   end if
end subroutine test_plan_nda

subroutine test_plan_cart(passed, failed)
   integer, intent(inout) :: passed, failed
   integer, parameter :: ndim = 3
   integer(c_int) :: dimensions(ndim), comm_dims(ndim)
   type(c_ptr) :: plan
   integer :: rank, nprocs
   integer(c_int) :: rc
   character(len=256) :: msg

   dimensions = [32, 32, 32]
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

   ! COMM_DIMS must have ndim elements, last must be 1
   comm_dims = [nprocs, 1, 1]

   if (rank == 0) write (*, '(A)', advance='no') '  shafftPlanCart                           '

   ! Fortran API: shafftPlanCart(plan, COMM_DIMS, dimensions, precision, comm)
   call shafftPlanCart(plan, comm_dims, dimensions, SHAFFT_C2C, MPI_COMM_WORLD, rc)

   if (rc == 0 .and. c_associated(plan)) then
      if (rank == 0) write (*, '(A)') 'PASS'
      passed = passed + 1
      call shafftDestroy(plan, rc)
   else
      if (rank == 0) then
         call shafftLastErrorMessage(msg)
         write (*, '(A,I0,A,A)') 'FAIL (rc=', rc, ', msg=', trim(msg), ')'
      end if
      failed = failed + 1
   end if
end subroutine test_plan_cart

subroutine test_plan_destroy(passed, failed)
   integer, intent(inout) :: passed, failed
   integer, parameter :: ndim = 2, nda = 1
   integer(c_int) :: dimensions(ndim)
   type(c_ptr) :: plan
   integer :: rank
   integer(c_int) :: rc
   character(len=256) :: msg

   dimensions = [16, 16]
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   if (rank == 0) write (*, '(A)', advance='no') '  shafftDestroy                            '

   call shafftPlanNDA(plan, ndim, nda, dimensions, SHAFFT_C2C, MPI_COMM_WORLD, rc)
   call shafftDestroy(plan, rc)

   ! After destroy, plan should be null
   if (.not. c_associated(plan) .and. rc == 0) then
      if (rank == 0) write (*, '(A)') 'PASS'
      passed = passed + 1
   else
      if (rank == 0) then
         call shafftLastErrorMessage(msg)
         write (*, '(A,I0,A,A)') 'FAIL (plan not nullified, rc=', rc, ', msg=', trim(msg), ')'
      end if
      failed = failed + 1
   end if
end subroutine test_plan_destroy

!============================================================================
! Buffer Management Tests
!============================================================================

subroutine test_alloc_free_buffer_sp(passed, failed)
   integer, intent(inout) :: passed, failed
   complex(c_float), pointer :: buf(:) => null()
   integer(c_size_t) :: count
   integer :: rank
   integer(c_int) :: rc
   character(len=256) :: msg

   count = 1024
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   if (rank == 0) write (*, '(A)', advance='no') '  shafftAllocBuffer_sp/FreeBuffer_sp       '

   call shafftAllocBuffer(count, buf, rc)

   if (rc == 0 .and. associated(buf)) then
      call shafftFreeBuffer(buf, rc)
      if (rank == 0) write (*, '(A)') 'PASS'
      passed = passed + 1
   else
      if (rank == 0) then
         call shafftLastErrorMessage(msg)
         write (*, '(A,I0,A,A)') 'FAIL (rc=', rc, ', msg=', trim(msg), ')'
      end if
      failed = failed + 1
   end if
end subroutine test_alloc_free_buffer_sp

subroutine test_alloc_free_buffer_dp(passed, failed)
   integer, intent(inout) :: passed, failed
   complex(c_double), pointer :: buf(:) => null()
   integer(c_size_t) :: count
   integer :: rank
   integer(c_int) :: rc
   character(len=256) :: msg

   count = 1024
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   if (rank == 0) write (*, '(A)', advance='no') '  shafftAllocBuffer_dp/FreeBuffer_dp       '

   call shafftAllocBuffer(count, buf, rc)

   if (rc == 0 .and. associated(buf)) then
      call shafftFreeBuffer(buf, rc)
      if (rank == 0) write (*, '(A)') 'PASS'
      passed = passed + 1
   else
      if (rank == 0) then
         call shafftLastErrorMessage(msg)
         write (*, '(A,I0,A,A)') 'FAIL (rc=', rc, ', msg=', trim(msg), ')'
      end if
      failed = failed + 1
   end if
end subroutine test_alloc_free_buffer_dp

subroutine test_copy_buffers_sp(passed, failed)
   integer, intent(inout) :: passed, failed
   complex(c_float), pointer :: device_buf(:) => null()
   complex(c_float), allocatable :: host_src(:), host_dst(:)
   integer(c_size_t) :: count
   integer :: i, rank
   integer(c_int) :: rc
   character(len=256) :: msg
   logical :: match

   count = 256
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   if (rank == 0) write (*, '(A)', advance='no') '  shafftCopyTo/FromBuffer_sp               '

   allocate (host_src(count))
   allocate (host_dst(count))
   do i = 1, int(count)
      host_src(i) = cmplx(real(i, c_float), real(i*2, c_float), kind=c_float)
   end do

   call shafftAllocBuffer(count, device_buf, rc); if (rc /= 0) goto 930
   call shafftCopyToBuffer(device_buf, host_src, count, rc); if (rc /= 0) goto 930
   call shafftCopyFromBuffer(host_dst, device_buf, count, rc); if (rc /= 0) goto 930

   match = .true.
   do i = 1, int(count)
      if (abs(real(host_src(i)) - real(host_dst(i))) > 1.0e-6) match = .false.
      if (abs(aimag(host_src(i)) - aimag(host_dst(i))) > 1.0e-6) match = .false.
   end do

   call shafftFreeBuffer(device_buf, rc)
   deallocate (host_src)
   deallocate (host_dst)

   if (match) then
      if (rank == 0) write (*, '(A)') 'PASS'
      passed = passed + 1
   else
      if (rank == 0) write (*, '(A)') 'FAIL (data mismatch)'
      failed = failed + 1
   end if
   return
930 continue
   if (rank == 0) then
      call shafftLastErrorMessage(msg)
      write (*, '(A,I0,A,A)') 'FAIL (rc=', rc, ', msg=', trim(msg), ')'
   end if
   failed = failed + 1
end subroutine test_copy_buffers_sp

subroutine test_copy_buffers_dp(passed, failed)
   integer, intent(inout) :: passed, failed
   complex(c_double), pointer :: device_buf(:) => null()
   complex(c_double), allocatable :: host_src(:), host_dst(:)
   integer(c_size_t) :: count
   integer :: i, rank
   integer(c_int) :: rc
   character(len=256) :: msg
   logical :: match

   count = 256
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   if (rank == 0) write (*, '(A)', advance='no') '  shafftCopyTo/FromBuffer_dp               '

   allocate (host_src(count))
   allocate (host_dst(count))
   do i = 1, int(count)
      host_src(i) = cmplx(real(i, c_double), real(i*2, c_double), kind=c_double)
   end do

   call shafftAllocBuffer(count, device_buf, rc); if (rc /= 0) goto 940
   call shafftCopyToBuffer(device_buf, host_src, count, rc); if (rc /= 0) goto 940
   call shafftCopyFromBuffer(host_dst, device_buf, count, rc); if (rc /= 0) goto 940

   match = .true.
   do i = 1, int(count)
      if (abs(real(host_src(i)) - real(host_dst(i))) > 1.0d-12) match = .false.
      if (abs(aimag(host_src(i)) - aimag(host_dst(i))) > 1.0d-12) match = .false.
   end do

   call shafftFreeBuffer(device_buf, rc)
   deallocate (host_src)
   deallocate (host_dst)

   if (match) then
      if (rank == 0) write (*, '(A)') 'PASS'
      passed = passed + 1
   else
      if (rank == 0) write (*, '(A)') 'FAIL (data mismatch)'
      failed = failed + 1
   end if
   return
940 continue
   if (rank == 0) then
      call shafftLastErrorMessage(msg)
      write (*, '(A,I0,A,A)') 'FAIL (rc=', rc, ', msg=', trim(msg), ')'
   end if
   failed = failed + 1
end subroutine test_copy_buffers_dp

!============================================================================
! Query Tests
!============================================================================

subroutine test_get_alloc_size(passed, failed)
   integer, intent(inout) :: passed, failed
   integer, parameter :: ndim = 3, nda = 1
   integer(c_int) :: dimensions(ndim)
   type(c_ptr) :: plan
   integer(c_size_t) :: alloc_size
   integer :: rank
   integer(c_int) :: rc
   character(len=256) :: msg

   dimensions = [32, 32, 32]
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   if (rank == 0) write (*, '(A)', advance='no') '  shafftGetAllocSize                       '

   call shafftPlanNDA(plan, ndim, nda, dimensions, SHAFFT_C2C, MPI_COMM_WORLD, rc); if (rc /= 0) goto 950
   call shafftGetAllocSize(plan, alloc_size, rc); if (rc /= 0) goto 950

   if (alloc_size > 0) then
      if (rank == 0) write (*, '(A)') 'PASS'
      passed = passed + 1
   else
      if (rank == 0) write (*, '(A)') 'FAIL (zero size)'
      failed = failed + 1
   end if

   call shafftDestroy(plan, rc)
   return
950 continue
   failed = failed + 1
   if (rank == 0) then
      call shafftLastErrorMessage(msg)
      write (*, '(A,I0,A,A)') 'FAIL (rc=', rc, ', msg=', trim(msg), ')'
   end if
end subroutine test_get_alloc_size

subroutine test_get_layout(passed, failed)
   integer, intent(inout) :: passed, failed
   integer, parameter :: ndim = 3, nda = 1
   integer(c_int) :: dimensions(ndim), subsize(ndim), offset(ndim)
   type(c_ptr) :: plan
   integer :: rank
   integer(c_int) :: rc
   character(len=256) :: msg

   dimensions = [32, 32, 32]
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   if (rank == 0) write (*, '(A)', advance='no') '  shafftGetLayout                          '

   call shafftPlanNDA(plan, ndim, nda, dimensions, SHAFFT_C2C, MPI_COMM_WORLD, rc); if (rc /= 0) goto 960
   call shafftGetLayout(plan, subsize, offset, SHAFFT_TENSOR_LAYOUT_INITIAL, rc); if (rc /= 0) goto 960

   if (subsize(1) > 0 .and. subsize(2) > 0 .and. subsize(3) > 0) then
      if (rank == 0) write (*, '(A)') 'PASS'
      passed = passed + 1
   else
      if (rank == 0) write (*, '(A)') 'FAIL (invalid subsize)'
      failed = failed + 1
   end if

   call shafftDestroy(plan, rc)
   return
960 continue
   failed = failed + 1
   if (rank == 0) then
      call shafftLastErrorMessage(msg)
      write (*, '(A,I0,A,A)') 'FAIL (rc=', rc, ', msg=', trim(msg), ')'
   end if
end subroutine test_get_layout

subroutine test_get_axes(passed, failed)
   integer, intent(inout) :: passed, failed
   integer, parameter :: ndim = 3, nda = 1
   integer(c_int) :: dimensions(ndim), ca(ndim), da(ndim)
   type(c_ptr) :: plan
   integer :: rank, nprocs
   integer(c_int) :: rc
   character(len=256) :: msg

   dimensions = [32, 32, 32]
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
   if (rank == 0) write (*, '(A)', advance='no') '  shafftGetAxes                            '

   ca = -1
   da = -1
   call shafftPlanNDA(plan, ndim, nda, dimensions, SHAFFT_C2C, MPI_COMM_WORLD, rc); if (rc /= 0) goto 970
   call shafftGetAxes(plan, ca, da, SHAFFT_TENSOR_LAYOUT_INITIAL, rc); if (rc /= 0) goto 970

   ! For nda=1 with multi-rank, da(1) should be 0 (first axis distributed)
   if (nprocs > 1) then
      if (da(1) == 0) then
         if (rank == 0) write (*, '(A)') 'PASS'
         passed = passed + 1
      else
         if (rank == 0) write (*, '(A)') 'FAIL (wrong distributed axis)'
         failed = failed + 1
      end if
   else
      ! Single rank - all contiguous
      if (rank == 0) write (*, '(A)') 'PASS'
      passed = passed + 1
   end if

   call shafftDestroy(plan, rc)
   return
970 continue
   failed = failed + 1
   if (rank == 0) then
      call shafftLastErrorMessage(msg)
      write (*, '(A,I0,A,A)') 'FAIL (rc=', rc, ', msg=', trim(msg), ')'
   end if
end subroutine test_get_axes

subroutine test_set_get_buffers(passed, failed)
   integer, intent(inout) :: passed, failed
   integer, parameter :: ndim = 2, nda = 1
   integer(c_int) :: dimensions(ndim)
   type(c_ptr) :: plan
   integer(c_size_t) :: alloc_size
   complex(c_float), pointer :: data_buf(:) => null()
   complex(c_float), pointer :: work_buf(:) => null()
   complex(c_float), pointer :: got_data(:) => null()
   complex(c_float), pointer :: got_work(:) => null()
   integer :: rank
   integer(c_int) :: rc
   character(len=256) :: msg

   dimensions = [16, 16]
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   if (rank == 0) write (*, '(A)', advance='no') '  shafftSet/GetBuffers                     '

   call shafftPlanNDA(plan, ndim, nda, dimensions, SHAFFT_C2C, MPI_COMM_WORLD, rc); if (rc /= 0) goto 980
   call shafftGetAllocSize(plan, alloc_size, rc); if (rc /= 0) goto 980

   call shafftAllocBuffer(alloc_size, data_buf, rc); if (rc /= 0) goto 980
   call shafftAllocBuffer(alloc_size, work_buf, rc); if (rc /= 0) goto 980
   call shafftSetBuffers(plan, data_buf, work_buf, rc); if (rc /= 0) goto 980
   call shafftGetBuffers(plan, alloc_size, got_data, got_work, rc); if (rc /= 0) goto 980

   ! Before execute, should get same pointers back
   if (associated(got_data) .and. associated(got_work)) then
      if (rank == 0) write (*, '(A)') 'PASS'
      passed = passed + 1
   else
      if (rank == 0) write (*, '(A)') 'FAIL (null pointers)'
      failed = failed + 1
   end if

   call shafftFreeBuffer(data_buf, rc)
   call shafftFreeBuffer(work_buf, rc)
   call shafftDestroy(plan, rc)
   return
980 continue
   failed = failed + 1
   if (rank == 0) then
      call shafftLastErrorMessage(msg)
      write (*, '(A,I0,A,A)') 'FAIL (rc=', rc, ', msg=', trim(msg), ')'
   end if
end subroutine test_set_get_buffers

!============================================================================
! Configuration Tests
!============================================================================

subroutine test_configuration_nda(passed, failed)
   integer, intent(inout) :: passed, failed
   integer, parameter :: ndim = 3
   integer(c_int) :: dimensions(ndim), subsize(ndim), offset(ndim), comm_dims(ndim)
   integer(c_int) :: nda
   integer :: rank, nprocs
   integer(c_int) :: rc
   character(len=256) :: msg

   dimensions = [64, 64, 32]
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
   if (rank == 0) write (*, '(A)', advance='no') '  shafftConfigurationNDA                   '

   nda = 1 ! request distribution (should fail on 1 rank)
   call shafftConfigurationNDA(ndim, dimensions, nda, subsize, offset, comm_dims, &
                               SHAFFT_C2C, int(0, c_size_t), MPI_COMM_WORLD, rc)

   if (nprocs == 1) then
      if (rc == SHAFFT_ERR_INVALID_DECOMP) then
         if (rank == 0) write (*, '(A)') 'PASS (expected INVALID_DECOMP on 1 rank)'
         passed = passed + 1
      else
         if (rank == 0) then
            call shafftLastErrorMessage(msg)
           write (*, '(A,I0,A,A)') 'FAIL (expected INVALID_DECOMP rc=', rc, ', msg=', trim(msg), ')'
         end if
         failed = failed + 1
      end if
   else
      if (rc == 0 .and. subsize(1) > 0 .and. subsize(2) > 0 .and. subsize(3) > 0) then
         if (rank == 0) write (*, '(A)') 'PASS'
         passed = passed + 1
      else
         if (rank == 0) then
            call shafftLastErrorMessage(msg)
            write (*, '(A,I0,A,A)') 'FAIL (rc=', rc, ', msg=', trim(msg), ')'
         end if
         failed = failed + 1
      end if
   end if
end subroutine test_configuration_nda

subroutine test_configuration_cart(passed, failed)
   integer, intent(inout) :: passed, failed
   integer, parameter :: ndim = 3
   integer(c_int) :: dimensions(ndim), subsize(ndim), offset(ndim), comm_dims(ndim)
   integer(c_int) :: comm_size
   integer :: rank, nprocs
   integer(c_int) :: rc
   character(len=256) :: msg

   dimensions = [64, 64, 32]
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

   ! COMM_DIMS must have ndim elements
   comm_dims = [nprocs, 1, 1]

   if (rank == 0) write (*, '(A)', advance='no') '  shafftConfigurationCart                  '

   call shafftConfigurationCart(ndim, dimensions, subsize, offset, comm_dims, comm_size, &
                                SHAFFT_C2C, int(0, c_size_t), MPI_COMM_WORLD, rc)

   if (rc == 0 .and. comm_size == nprocs) then
      if (rank == 0) write (*, '(A)') 'PASS'
      passed = passed + 1
   else
      if (rank == 0) then
         call shafftLastErrorMessage(msg)
         write (*, '(A,I0,A,A)') 'FAIL (rc=', rc, ', msg=', trim(msg), ')'
      end if
      failed = failed + 1
   end if
end subroutine test_configuration_cart

end program test_f03_api
