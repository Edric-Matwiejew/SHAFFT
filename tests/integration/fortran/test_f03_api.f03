!> @file test_f03_api.f03
!! @brief Comprehensive Fortran API coverage test
program test_f03_api
use iso_c_binding
use mpi
use shafft

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
call test_last_error_source(passed, failed)
call test_last_error_domain_code(passed, failed)

! Plan lifecycle tests
call test_plan_nda(passed, failed)
call test_plan_cart(passed, failed)
call test_plan_destroy(passed, failed)
call test_plan_output_policy(passed, failed)

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
call test_configuration_1d(passed, failed)

! Legacy 1D plan lifecycle
call test_1d_init(passed, failed)

! Library teardown
call test_finalize(passed, failed)

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
   type(c_ptr) :: tmpPlan
   integer(c_size_t) :: tmpSize
   integer(c_int) :: rc

   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   if (rank == 0) write (*, '(A)', advance='no') '  shafftClearLastError                     '

   ! First trigger an error so there is something to clear.
   ! Calling getAllocSize on a created-but-uninitialized plan fires
   ! SHAFFT_FAIL(SHAFFT_ERR_PLAN_NOT_INIT) which sets the last error.
   call shafftNDCreate(tmpPlan, rc)
   if (rc /= 0) then
      if (rank == 0) write (*, '(A)') 'FAIL (could not create plan)'
      failed = failed + 1
      return
   end if
   call shafftGetAllocSize(tmpPlan, tmpSize, rc) ! rc should be non-zero
   call shafftDestroy(tmpPlan, rc)

   status = shafftLastErrorStatus()
   if (status == 0) then
      if (rank == 0) write (*, '(A)') 'FAIL (expected non-zero status before clear)'
      failed = failed + 1
      return
   end if

   ! Now clear and verify
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
   integer, parameter :: ndim = 3
   integer(c_int) :: dimensions(ndim), commDims(ndim)
   type(c_ptr) :: plan
   integer :: rank, nprocs
   integer(c_int) :: rc
   character(len=256) :: msg

   dimensions = [32, 32, 32]
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

   ! commDims for slab decomposition (like nda=1): [nprocs, 1, 1]
   commDims = [nprocs, 1, 1]

   if (rank == 0) write (*, '(A)', advance='no') '  shafftNDInit (slab)                      '

   call shafftNDCreate(plan, rc)
     if (rc /= 0) then
       call shafftLastErrorMessage(msg)
       if (rank == 0) write (*, '(A,I0,3A)') &
            'FAIL (create failed, rc=', rc, ', msg=', trim(msg), ')'
       failed = failed + 1
       return
     end if
   call shafftNDInit(plan, commDims, dimensions, SHAFFT_C2C, MPI_COMM_WORLD, &
                     SHAFFT_LAYOUT_REDISTRIBUTED, rc)

   if (rc == 0 .and. c_associated(plan)) then
      if (rank == 0) write (*, '(A)') 'PASS'
      passed = passed + 1
      call shafftDestroy(plan, rc)
   else
      if (rank == 0) then
         call shafftLastErrorMessage(msg)
         write (*, '(A,I0,3A)') 'FAIL (rc=', rc, ', msg=', trim(msg), ')'
      end if
      failed = failed + 1
      if (c_associated(plan)) call shafftDestroy(plan, rc)
   end if
end subroutine test_plan_nda

subroutine test_plan_cart(passed, failed)
   integer, intent(inout) :: passed, failed
   integer, parameter :: ndim = 2
   integer(c_int) :: dimensions(ndim), commDims(ndim)
   type(c_ptr) :: plan
   integer :: rank, nprocs
   integer(c_int) :: rc
   character(len=256) :: msg

   dimensions = [64, 32]
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

   ! 2D slab decomposition (exercises a different ndim than test_plan_nda)
   commDims = [nprocs, 1]

   if (rank == 0) write (*, '(A)', advance='no') '  shafftNDInit (2D slab)                   '

   ! Fortran API: shafftNDInit(plan, commDims, dimensions, precision, comm, outputPolicy, ierr)
   call shafftNDCreate(plan, rc)
     if (rc /= 0) then
       call shafftLastErrorMessage(msg)
       if (rank == 0) write (*, '(A,I0,3A)') &
            'FAIL (create failed, rc=', rc, ', msg=', trim(msg), ')'
       failed = failed + 1
       return
     end if
   call shafftNDInit(plan, commDims, dimensions, SHAFFT_C2C, MPI_COMM_WORLD, &
                     SHAFFT_LAYOUT_REDISTRIBUTED, rc)

   if (rc == 0 .and. c_associated(plan)) then
      if (rank == 0) write (*, '(A)') 'PASS'
      passed = passed + 1
      call shafftDestroy(plan, rc)
   else
      if (rank == 0) then
         call shafftLastErrorMessage(msg)
         write (*, '(A,I0,3A)') 'FAIL (rc=', rc, ', msg=', trim(msg), ')'
      end if
      failed = failed + 1
      if (c_associated(plan)) call shafftDestroy(plan, rc)
   end if
end subroutine test_plan_cart

subroutine test_plan_destroy(passed, failed)
   integer, intent(inout) :: passed, failed
   integer, parameter :: ndim = 2
   integer(c_int) :: dimensions(ndim), commDims(ndim)
   type(c_ptr) :: plan
   integer :: rank, nprocs
   integer(c_int) :: rc
   character(len=256) :: msg

   dimensions = [16, 16]
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

   commDims = [nprocs, 1]

   if (rank == 0) write (*, '(A)', advance='no') '  shafftDestroy                            '

   call shafftNDCreate(plan, rc)
     if (rc /= 0) then
       call shafftLastErrorMessage(msg)
       if (rank == 0) write (*, '(A,I0,3A)') &
            'FAIL (create failed, rc=', rc, ', msg=', trim(msg), ')'
       failed = failed + 1
       return
     end if
   call shafftNDInit(plan, commDims, dimensions, SHAFFT_C2C, MPI_COMM_WORLD, &
                     SHAFFT_LAYOUT_REDISTRIBUTED, rc)
   call shafftDestroy(plan, rc)

   ! After destroy, plan should be null
   if (.not. c_associated(plan) .and. rc == 0) then
      if (rank == 0) write (*, '(A)') 'PASS'
      passed = passed + 1
   else
      if (rank == 0) then
         call shafftLastErrorMessage(msg)
         write (*, '(A,I0,3A)') 'FAIL (plan not nullified, rc=', rc, ', msg=', trim(msg), ')'
      end if
      failed = failed + 1
   end if
end subroutine test_plan_destroy

subroutine test_plan_output_policy(passed, failed)
   integer, intent(inout) :: passed, failed
   integer, parameter :: ndim = 2
   integer(c_int) :: dimensions(ndim), commDims(ndim)
   integer(c_size_t) :: localAllocSize
   integer(c_size_t) :: init_subsize(ndim), init_offset(ndim)
   integer(c_size_t) :: cur_subsize(ndim), cur_offset(ndim)
   type(c_ptr) :: plan = c_null_ptr
   complex(c_float), pointer :: data_buf(:) => null()
   complex(c_float), pointer :: work_buf(:) => null()
   integer :: rank, nprocs
   integer(c_int) :: rc
   character(len=256) :: msg

   dimensions = [16, 16]
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
   commDims = [nprocs, 1]

   if (rank == 0) write (*, '(A)', advance='no') '  shafftNDInit(outputPolicy=INITIAL)       '

   call shafftNDCreate(plan, rc); if (rc /= 0) goto 985
   call shafftNDInit(plan, commDims, dimensions, SHAFFT_C2C, MPI_COMM_WORLD, &
                     SHAFFT_LAYOUT_INITIAL, rc)
   if (rc /= 0) goto 985
   call shafftGetAllocSize(plan, localAllocSize, rc); if (rc /= 0) goto 985
   call shafftAllocBuffer(localAllocSize, data_buf, rc); if (rc /= 0) goto 985
   call shafftAllocBuffer(localAllocSize, work_buf, rc); if (rc /= 0) goto 985
   call shafftSetBuffers(plan, data_buf, work_buf, rc); if (rc /= 0) goto 985
   call shafftPlan(plan, rc); if (rc /= 0) goto 985
   call shafftGetLayout(plan, init_subsize, init_offset, SHAFFT_TENSOR_LAYOUT_INITIAL, rc)
   if (rc /= 0) goto 985

   call shafftExecute(plan, SHAFFT_FORWARD, rc); if (rc /= 0) goto 985
   call shafftGetLayout(plan, cur_subsize, cur_offset, SHAFFT_TENSOR_LAYOUT_CURRENT, rc)
   if (rc /= 0) goto 985

   if (all(init_subsize == cur_subsize) .and. all(init_offset == cur_offset)) then
      if (rank == 0) write (*, '(A)') 'PASS'
      passed = passed + 1
   else
      if (rank == 0) write (*, '(A)') 'FAIL (current layout is not initial after forward)'
      failed = failed + 1
   end if

   call shafftFreeBuffer(data_buf, rc)
   call shafftFreeBuffer(work_buf, rc)
   call shafftDestroy(plan, rc)
   return
985 continue
   failed = failed + 1
   if (rank == 0) then
      call shafftLastErrorMessage(msg)
      write (*, '(A,I0,3A)') 'FAIL (rc=', rc, ', msg=', trim(msg), ')'
   end if
   if (associated(data_buf)) call shafftFreeBuffer(data_buf, rc)
   if (associated(work_buf)) call shafftFreeBuffer(work_buf, rc)
   if (c_associated(plan)) call shafftDestroy(plan, rc)
end subroutine test_plan_output_policy

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
         write (*, '(A,I0,3A)') 'FAIL (rc=', rc, ', msg=', trim(msg), ')'
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
         write (*, '(A,I0,3A)') 'FAIL (rc=', rc, ', msg=', trim(msg), ')'
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
      write (*, '(A,I0,3A)') 'FAIL (rc=', rc, ', msg=', trim(msg), ')'
   end if
   failed = failed + 1
   if (associated(device_buf)) call shafftFreeBuffer(device_buf, rc)
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
      write (*, '(A,I0,3A)') 'FAIL (rc=', rc, ', msg=', trim(msg), ')'
   end if
   failed = failed + 1
   if (associated(device_buf)) call shafftFreeBuffer(device_buf, rc)
end subroutine test_copy_buffers_dp

!============================================================================
! Query Tests
!============================================================================

subroutine test_get_alloc_size(passed, failed)
   integer, intent(inout) :: passed, failed
   integer, parameter :: ndim = 3
   integer(c_int) :: dimensions(ndim), commDims(ndim)
   type(c_ptr) :: plan = c_null_ptr
   integer(c_size_t) :: localAllocSize
   integer :: rank, nprocs
   integer(c_int) :: rc
   character(len=256) :: msg

   dimensions = [32, 32, 32]
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

   commDims = [nprocs, 1, 1]

   if (rank == 0) write (*, '(A)', advance='no') '  shafftGetAllocSize                       '

   call shafftNDCreate(plan, rc); if (rc /= 0) goto 950
   call shafftNDInit(plan, commDims, dimensions, SHAFFT_C2C, MPI_COMM_WORLD, &
                     SHAFFT_LAYOUT_REDISTRIBUTED, rc)
   if (rc /= 0) goto 950
   call shafftGetAllocSize(plan, localAllocSize, rc); if (rc /= 0) goto 950

   if (localAllocSize > 0) then
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
      write (*, '(A,I0,3A)') 'FAIL (rc=', rc, ', msg=', trim(msg), ')'
   end if
   if (c_associated(plan)) call shafftDestroy(plan, rc)
end subroutine test_get_alloc_size

subroutine test_get_layout(passed, failed)
   integer, intent(inout) :: passed, failed
   integer, parameter :: ndim = 3
   integer(c_int) :: dimensions(ndim), commDims(ndim)
   integer(c_size_t) :: subsize(ndim), offset(ndim)
   type(c_ptr) :: plan = c_null_ptr
   integer :: rank, nprocs
   integer(c_int) :: rc
   character(len=256) :: msg

   dimensions = [32, 32, 32]
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

   commDims = [nprocs, 1, 1]

   if (rank == 0) write (*, '(A)', advance='no') '  shafftGetLayout                          '

   call shafftNDCreate(plan, rc); if (rc /= 0) goto 960
   call shafftNDInit(plan, commDims, dimensions, SHAFFT_C2C, MPI_COMM_WORLD, &
                     SHAFFT_LAYOUT_REDISTRIBUTED, rc)
   if (rc /= 0) goto 960
   call shafftGetLayout(plan, subsize, offset, SHAFFT_TENSOR_LAYOUT_INITIAL, rc)
   if (rc /= 0) goto 960

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
      write (*, '(A,I0,3A)') 'FAIL (rc=', rc, ', msg=', trim(msg), ')'
   end if
   if (c_associated(plan)) call shafftDestroy(plan, rc)
end subroutine test_get_layout

subroutine test_get_axes(passed, failed)
   integer, intent(inout) :: passed, failed
   integer, parameter :: ndim = 3
   integer(c_int) :: dimensions(ndim), commDims(ndim), ca(ndim), da(ndim)
   type(c_ptr) :: plan = c_null_ptr
   integer :: rank, nprocs
   integer(c_int) :: rc
   character(len=256) :: msg

   dimensions = [32, 32, 32]
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

   commDims = [nprocs, 1, 1]

   if (rank == 0) write (*, '(A)', advance='no') '  shafftGetAxes                            '

   ca = -1
   da = -1
   call shafftNDCreate(plan, rc); if (rc /= 0) goto 970
   call shafftNDInit(plan, commDims, dimensions, SHAFFT_C2C, MPI_COMM_WORLD, &
                     SHAFFT_LAYOUT_REDISTRIBUTED, rc)
   if (rc /= 0) goto 970
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
      write (*, '(A,I0,3A)') 'FAIL (rc=', rc, ', msg=', trim(msg), ')'
   end if
   if (c_associated(plan)) call shafftDestroy(plan, rc)
end subroutine test_get_axes

subroutine test_set_get_buffers(passed, failed)
   integer, intent(inout) :: passed, failed
   integer, parameter :: ndim = 2
   integer(c_int) :: dimensions(ndim), commDims(ndim)
   type(c_ptr) :: plan = c_null_ptr
   integer(c_size_t) :: localAllocSize
   complex(c_float), pointer :: data_buf(:) => null()
   complex(c_float), pointer :: work_buf(:) => null()
   complex(c_float), pointer :: got_data(:) => null()
   complex(c_float), pointer :: got_work(:) => null()
   integer :: rank, nprocs
   integer(c_int) :: rc
   character(len=256) :: msg

   dimensions = [16, 16]
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

   commDims = [nprocs, 1]

   if (rank == 0) write (*, '(A)', advance='no') '  shafftSet/GetBuffers                     '

   call shafftNDCreate(plan, rc); if (rc /= 0) goto 980
   call shafftNDInit(plan, commDims, dimensions, SHAFFT_C2C, MPI_COMM_WORLD, &
                     SHAFFT_LAYOUT_REDISTRIBUTED, rc)
   if (rc /= 0) goto 980
   call shafftGetAllocSize(plan, localAllocSize, rc); if (rc /= 0) goto 980

   call shafftAllocBuffer(localAllocSize, data_buf, rc); if (rc /= 0) goto 980
   call shafftAllocBuffer(localAllocSize, work_buf, rc); if (rc /= 0) goto 980
   call shafftSetBuffers(plan, data_buf, work_buf, rc); if (rc /= 0) goto 980
   call shafftGetBuffers(plan, localAllocSize, got_data, got_work, rc); if (rc /= 0) goto 980

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
      write (*, '(A,I0,3A)') 'FAIL (rc=', rc, ', msg=', trim(msg), ')'
   end if
   if (associated(data_buf)) call shafftFreeBuffer(data_buf, rc)
   if (associated(work_buf)) call shafftFreeBuffer(work_buf, rc)
   if (c_associated(plan)) call shafftDestroy(plan, rc)
end subroutine test_set_get_buffers

!============================================================================
! Configuration Tests
!============================================================================

subroutine test_configuration_nda(passed, failed)
   integer, intent(inout) :: passed, failed
   integer, parameter :: ndim = 3
   integer(c_int) :: dimensions(ndim), commDims(ndim)
   integer(c_size_t) :: subsize(ndim), offset(ndim)
   integer(c_int) :: nda, commSize
   integer :: rank, nprocs
   integer(c_int) :: rc
   character(len=256) :: msg

   dimensions = [64, 64, 32]
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
   if (rank == 0) write (*, '(A)', advance='no') '  shafftConfigurationND (maximize)         '

   ! Use SHAFFT_MAXIMIZE_NDA strategy with nda=1 (slab decomposition)
   nda = 1
   commDims = [0, 0, 0]  ! Let shafft compute optimal decomposition
   call shafftConfigurationND(dimensions, SHAFFT_C2C, commDims, nda, subsize, offset, &
                              commSize, SHAFFT_MAXIMIZE_NDA, int(0, c_size_t), MPI_COMM_WORLD, rc)

   if (nprocs == 1) then
      ! On 1 rank, MAXIMIZE_NDA strategy may either:
      ! - Return success with degenerate decomposition (nda may be adjusted to 0)
      ! - Return INVALID_DECOMP
      ! Either is acceptable behavior for the unified API
      if (rc == 0 .or. rc == SHAFFT_ERR_INVALID_DECOMP) then
         if (rank == 0) write (*, '(A)') 'PASS'
         passed = passed + 1
      else
         if (rank == 0) then
            call shafftLastErrorMessage(msg)
           write (*, '(A,I0,3A)') 'FAIL (unexpected error rc=', rc, ', msg=', trim(msg), ')'
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
            write (*, '(A,I0,3A)') 'FAIL (rc=', rc, ', msg=', trim(msg), ')'
         end if
         failed = failed + 1
      end if
   end if
end subroutine test_configuration_nda

subroutine test_configuration_cart(passed, failed)
   integer, intent(inout) :: passed, failed
   integer, parameter :: ndim = 3
   integer(c_int) :: dimensions(ndim), commDims(ndim)
   integer(c_size_t) :: subsize(ndim), offset(ndim)
   integer(c_int) :: commSize, nda
   integer :: rank, nprocs
   integer(c_int) :: rc
   character(len=256) :: msg

   dimensions = [64, 64, 32]
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

   ! commDims must have ndim elements - specify explicit decomposition
   commDims = [nprocs, 1, 1]
   nda = 1

   if (rank == 0) write (*, '(A)', advance='no') '  shafftConfigurationND (explicit)         '

   call shafftConfigurationND(dimensions, SHAFFT_C2C, commDims, nda, subsize, offset, &
                              commSize, SHAFFT_MAXIMIZE_NDA, int(0, c_size_t), MPI_COMM_WORLD, rc)

   if (rc == 0 .and. commSize == nprocs) then
      if (rank == 0) write (*, '(A)') 'PASS'
      passed = passed + 1
   else
      if (rank == 0) then
         call shafftLastErrorMessage(msg)
         write (*, '(A,I0,3A)') 'FAIL (rc=', rc, ', msg=', trim(msg), ')'
      end if
      failed = failed + 1
   end if
end subroutine test_configuration_cart

!============================================================================
! Legacy 1D Configuration / Plan Tests
!============================================================================

subroutine test_configuration_1d(passed, failed)
   integer, intent(inout) :: passed, failed
   integer(c_size_t) :: N, localN, localStart, localAllocSize
   integer :: rank
   integer(c_int) :: rc

   N = 1024
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   if (rank == 0) write (*, '(A)', advance='no') '  shafftConfiguration1D                    '

   call shafftConfiguration1D(N, localN, localStart, localAllocSize, &
                              SHAFFT_C2C, MPI_COMM_WORLD, rc)

   if (rc == 0 .and. localN > 0 .and. localAllocSize >= localN .and. localStart >= 0) then
      if (rank == 0) write (*, '(A)') 'PASS'
      passed = passed + 1
   else
      if (rank == 0) write (*, '(A,I0,A,I0,A,I0,A,I0,A)') &
            'FAIL (rc=', rc, ', localN=', localN, ', localStart=', localStart, &
            ', allocSize=', localAllocSize, ')'
      failed = failed + 1
   end if
end subroutine test_configuration_1d

subroutine test_1d_init(passed, failed)
   integer, intent(inout) :: passed, failed
   type(c_ptr) :: plan = c_null_ptr
   integer(c_size_t) :: N, localN, localStart, localAllocSize, allocSize
   integer :: rank
   integer(c_int) :: rc
   character(len=256) :: msg

   N = 1024
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   if (rank == 0) write (*, '(A)', advance='no') '  shafft1DInit lifecycle                   '

   ! Step 1: query decomposition
   call shafftConfiguration1D(N, localN, localStart, localAllocSize, &
                              SHAFFT_C2C, MPI_COMM_WORLD, rc)
   if (rc /= 0) goto 910

   ! Step 2: create + init
   call shafft1DCreate(plan, rc)
   if (rc /= 0) goto 910

   call shafft1DInit(plan, N, localN, localStart, SHAFFT_C2C, MPI_COMM_WORLD, rc)
   if (rc /= 0) goto 910

   ! Step 3: plan
   call shafftPlan(plan, rc)
   if (rc /= 0) goto 910

   ! Step 4: verify alloc size > 0
   call shafftGetAllocSize(plan, allocSize, rc)
   if (rc /= 0 .or. allocSize == 0) goto 910

   ! Success
   if (rank == 0) write (*, '(A)') 'PASS'
   passed = passed + 1
   call shafftDestroy(plan, rc)
   return

910 continue
   failed = failed + 1
   if (rank == 0) then
      call shafftLastErrorMessage(msg)
      write (*, '(3A)') 'FAIL (', trim(msg), ')'
   end if
   if (c_associated(plan)) call shafftDestroy(plan, rc)
end subroutine test_1d_init

!============================================================================
! Additional Error API Tests
!============================================================================

subroutine test_last_error_source(passed, failed)
   integer, intent(inout) :: passed, failed
   integer :: source, rank
   type(c_ptr) :: tmpPlan
   integer(c_size_t) :: tmpSize
   integer(c_int) :: rc

   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   if (rank == 0) write (*, '(A)', advance='no') '  shafftLastErrorSource                    '

   ! Trigger a SHAFFT-internal error (source = SHAFFT_ERRSRC_NONE, code = 0)
   call shafftNDCreate(tmpPlan, rc)
   if (rc /= 0) then
      if (rank == 0) write (*, '(A)') 'FAIL (could not create plan)'
      failed = failed + 1
      return
   end if
   call shafftGetAllocSize(tmpPlan, tmpSize, rc) ! ERR_PLAN_NOT_INIT
   call shafftDestroy(tmpPlan, rc)

   ! SHAFFT-internal errors use source=NONE
   source = shafftLastErrorSource()
   if (source /= SHAFFT_ERRSRC_NONE) then
      if (rank == 0) write (*, '(A,I0,A)') 'FAIL (source=', source, &
            ', expected ERRSRC_NONE after SHAFFT error)'
      failed = failed + 1
      return
   end if

   ! After clear, source should still be NONE
   call shafftClearLastError()
   source = shafftLastErrorSource()

   if (source == SHAFFT_ERRSRC_NONE) then
      if (rank == 0) write (*, '(A)') 'PASS'
      passed = passed + 1
   else
      if (rank == 0) write (*, '(A,I0,A)') 'FAIL (source=', source, ', expected 0 after clear)'
      failed = failed + 1
   end if
end subroutine test_last_error_source

subroutine test_last_error_domain_code(passed, failed)
   integer, intent(inout) :: passed, failed
   integer :: code, status, rank
   type(c_ptr) :: tmpPlan
   integer(c_size_t) :: tmpSize
   integer(c_int) :: rc

   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   if (rank == 0) write (*, '(A)', advance='no') '  shafftLastErrorDomainCode                '

   ! Trigger a SHAFFT-internal error (domain code = 0 for internal errors)
   call shafftNDCreate(tmpPlan, rc)
   if (rc /= 0) then
      if (rank == 0) write (*, '(A)') 'FAIL (could not create plan)'
      failed = failed + 1
      return
   end if
   call shafftGetAllocSize(tmpPlan, tmpSize, rc) ! ERR_PLAN_NOT_INIT
   call shafftDestroy(tmpPlan, rc)

   ! Verify error was recorded
   status = shafftLastErrorStatus()
   if (status == 0) then
      if (rank == 0) write (*, '(A)') 'FAIL (expected non-zero status)'
      failed = failed + 1
      return
   end if

   ! SHAFFT-internal errors have domain code = 0
   code = shafftLastErrorDomainCode()
   if (code /= 0) then
      if (rank == 0) write (*, '(A,I0,A)') 'FAIL (code=', code, ', expected 0 for internal error)'
      failed = failed + 1
      return
   end if

   ! After clear, domain code should remain 0
   call shafftClearLastError()
   code = shafftLastErrorDomainCode()

   if (code == 0) then
      if (rank == 0) write (*, '(A)') 'PASS'
      passed = passed + 1
   else
      if (rank == 0) write (*, '(A,I0,A)') 'FAIL (code=', code, ', expected 0 after clear)'
      failed = failed + 1
   end if
end subroutine test_last_error_domain_code

!============================================================================
! Finalize Test
!============================================================================

subroutine test_finalize(passed, failed)
   integer, intent(inout) :: passed, failed
   integer :: rank
   integer(c_int) :: rc

   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   if (rank == 0) write (*, '(A)', advance='no') '  shafftFinalize                           '

   ! First call should succeed
   call shafftFinalize(rc)
   if (rc /= 0) then
      if (rank == 0) write (*, '(A,I0,A)') 'FAIL (rc=', rc, ')'
      failed = failed + 1
      return
   end if

   ! Second call should also succeed (idempotent)
   call shafftFinalize(rc)
   if (rc /= 0) then
      if (rank == 0) write (*, '(A,I0,A)') 'FAIL (2nd call rc=', rc, ')'
      failed = failed + 1
      return
   end if

   if (rank == 0) write (*, '(A)') 'PASS'
   passed = passed + 1
end subroutine test_finalize

end program test_f03_api
