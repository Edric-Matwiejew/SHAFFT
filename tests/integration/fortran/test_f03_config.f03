!> @file test_f03_config.f03
!! @brief Fortran tests for the config-driven plan initialisation API.
!!
!! Covers: shafft_config_nd_t / shafft_config_1d_t derived types,
!!         shafftConfigNDInit / Release / Resolve / NDInitFromConfig,
!!         shafftConfig1DInit / Release / Resolve / 1DInitFromConfig,
!!         shafftGetCommunicator.
program test_f03_config
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
   write (*, '(A)') '=== Fortran Config API Tests ==='
end if

! ND config tests
call test_config_nd_lifecycle(passed, failed)
call test_config_nd_fields(passed, failed)
call test_config_nd_resolve(passed, failed)
call test_config_nd_init_from_config(passed, failed)

! 1D config tests
call test_config_1d_lifecycle(passed, failed)
call test_config_1d_fields(passed, failed)
call test_config_1d_resolve(passed, failed)
call test_config_1d_init_from_config(passed, failed)

! Communicator query
call test_get_communicator_nd(passed, failed)
call test_get_communicator_1d(passed, failed)

if (rank == 0) then
   write (*, '(A)') ''
   if (failed == 0) then
      write (*, '(A)') '=== Fortran Config API Tests: PASSED ==='
   else
      write (*, '(A)') '=== Fortran Config API Tests: FAILED ==='
   end if
   write (*, '(A,I0,A,I0)') 'Passed: ', passed, ', Failed: ', failed
end if

call MPI_Finalize(ierr)

if (failed > 0) then
   call exit(1)
end if

contains

!============================================================================
! N-D Config Tests
!============================================================================

subroutine test_config_nd_lifecycle(passed, failed)
   integer, intent(inout) :: passed, failed
   type(shafft_config_nd_t) :: cfg
   integer, parameter :: ndim = 3
   integer(c_size_t) :: globalShape(ndim)
   integer(c_int) :: rc

   globalShape = [32, 32, 32]
   if (rank == 0) write (*, '(A)', advance='no') '  configND init/release                    '

   call shafftConfigNDInit(cfg, ndim, globalShape, SHAFFT_C2C, &
                           hintNda=0, strategy=0, &
                           outputPolicy=SHAFFT_LAYOUT_REDISTRIBUTED, &
                           memLimit=0_c_size_t, comm=MPI_COMM_WORLD, ierr=rc)

   if (rc /= SHAFFT_SUCCESS) then
      if (rank == 0) write (*, '(A,I0,A)') 'FAIL (init rc=', rc, ')'
      failed = failed + 1
      return
   end if

   ! Handle should be non-null after init
   if (.not. c_associated(cfg%handle)) then
      if (rank == 0) write (*, '(A)') 'FAIL (handle null after init)'
      failed = failed + 1
      call shafftConfigNDRelease(cfg)
      return
   end if

   call shafftConfigNDRelease(cfg)

   ! Handle should be null after release
   if (c_associated(cfg%handle)) then
      if (rank == 0) write (*, '(A)') 'FAIL (handle non-null after release)'
      failed = failed + 1
      return
   end if

   if (rank == 0) write (*, '(A)') 'PASS'
   passed = passed + 1
end subroutine test_config_nd_lifecycle

subroutine test_config_nd_fields(passed, failed)
   integer, intent(inout) :: passed, failed
   type(shafft_config_nd_t) :: cfg
   integer, parameter :: ndim = 3
   integer(c_size_t) :: globalShape(ndim)
   integer(c_int) :: rc
   character(len=256) :: msg
   logical :: ok

   globalShape = [16, 16, 16]
   if (rank == 0) write (*, '(A)', advance='no') '  configND resolved fields                 '

   call shafftConfigNDInit(cfg, ndim, globalShape, SHAFFT_C2C, &
                           hintNda=0, strategy=0, &
                           outputPolicy=SHAFFT_LAYOUT_REDISTRIBUTED, &
                           memLimit=0_c_size_t, comm=MPI_COMM_WORLD, ierr=rc)
   if (rc /= SHAFFT_SUCCESS) goto 990

   ok = .true.

   ! Check basic resolved fields
   if (cfg%ndim_ /= ndim) then
      if (rank == 0) write (*, '(A,I0,A)') 'FAIL (ndim=', cfg%ndim_, ')'
      ok = .false.
   end if

   if (cfg%precision_ /= SHAFFT_C2C) then
      if (rank == 0) write (*, '(A,I0,A)') 'FAIL (precision=', cfg%precision_, ')'
      ok = .false.
   end if

   if (cfg%allocElements <= 0) then
      if (rank == 0) write (*, '(A,I0,A)') 'FAIL (allocElements=', cfg%allocElements, ')'
      ok = .false.
   end if

   if (cfg%isActive /= 1 .and. cfg%isActive /= 0) then
      if (rank == 0) write (*, '(A,I0,A)') 'FAIL (isActive=', cfg%isActive, ')'
      ok = .false.
   end if

   if (cfg%commSize_ /= nprocs) then
      if (rank == 0) write (*, '(A,I0,A)') 'FAIL (commSize=', cfg%commSize_, ')'
      ok = .false.
   end if

   ! RESOLVED flag must be set
   if (iand(cfg%flags, SHAFFT_CONFIG_RESOLVED) == 0) then
      if (rank == 0) write (*, '(A,I0,A)') 'FAIL (flags=', cfg%flags, ', missing RESOLVED)'
      ok = .false.
   end if

   ! globalShape should match input
   if (.not. allocated(cfg%globalShape)) then
      if (rank == 0) write (*, '(A)') 'FAIL (globalShape not allocated)'
      ok = .false.
   else if (size(cfg%globalShape) /= ndim) then
      if (rank == 0) write (*, '(A)') 'FAIL (globalShape wrong size)'
      ok = .false.
   else if (any(cfg%globalShape /= globalShape)) then
      if (rank == 0) write (*, '(A)') 'FAIL (globalShape mismatch)'
      ok = .false.
   end if

   ! commDims should be allocated with ndim elements
   if (.not. allocated(cfg%commDims)) then
      if (rank == 0) write (*, '(A)') 'FAIL (commDims not allocated)'
      ok = .false.
   else if (size(cfg%commDims) /= ndim) then
      if (rank == 0) write (*, '(A)') 'FAIL (commDims wrong size)'
      ok = .false.
   end if

   if (ok) then
      if (rank == 0) write (*, '(A)') 'PASS'
      passed = passed + 1
   else
      failed = failed + 1
   end if

   call shafftConfigNDRelease(cfg)
   return

990 continue
   failed = failed + 1
   if (rank == 0) then
      call shafftLastErrorMessage(msg)
      write (*, '(A,I0,3A)') 'FAIL (init rc=', rc, ', msg=', trim(msg), ')'
   end if
end subroutine test_config_nd_fields

subroutine test_config_nd_resolve(passed, failed)
   integer, intent(inout) :: passed, failed
   type(shafft_config_nd_t) :: cfg
   integer, parameter :: ndim = 2
   integer(c_size_t) :: globalShape(ndim)
   integer(c_int) :: rc
   character(len=256) :: msg
   integer :: origPolicy

   globalShape = [64, 64]
   if (rank == 0) write (*, '(A)', advance='no') '  configND resolve (change outputPolicy)   '

   call shafftConfigNDInit(cfg, ndim, globalShape, SHAFFT_C2C, &
                           hintNda=0, strategy=0, &
                           outputPolicy=SHAFFT_LAYOUT_REDISTRIBUTED, &
                           memLimit=0_c_size_t, comm=MPI_COMM_WORLD, ierr=rc)
   if (rc /= SHAFFT_SUCCESS) goto 991

   origPolicy = cfg%outputPolicy

   ! Change outputPolicy and re-resolve
   cfg%outputPolicy = SHAFFT_LAYOUT_INITIAL
   call shafftConfigNDResolve(cfg, rc)
   if (rc /= SHAFFT_SUCCESS) goto 991

   if (cfg%outputPolicy /= SHAFFT_LAYOUT_INITIAL) then
      if (rank == 0) write (*, '(A,I0,A)') 'FAIL (outputPolicy=', cfg%outputPolicy, ')'
      failed = failed + 1
      call shafftConfigNDRelease(cfg)
      return
   end if

   if (rank == 0) write (*, '(A)') 'PASS'
   passed = passed + 1
   call shafftConfigNDRelease(cfg)
   return

991 continue
   failed = failed + 1
   if (rank == 0) then
      call shafftLastErrorMessage(msg)
      write (*, '(A,I0,3A)') 'FAIL (rc=', rc, ', msg=', trim(msg), ')'
   end if
   if (c_associated(cfg%handle)) call shafftConfigNDRelease(cfg)
end subroutine test_config_nd_resolve

subroutine test_config_nd_init_from_config(passed, failed)
   integer, intent(inout) :: passed, failed
   type(shafft_config_nd_t) :: cfg
   type(c_ptr) :: plan = c_null_ptr
   integer, parameter :: ndim = 3
   integer(c_size_t) :: globalShape(ndim)
   integer(c_size_t) :: allocSize
   integer(c_int) :: rc
   character(len=256) :: msg

   globalShape = [16, 16, 16]
   if (rank == 0) write (*, '(A)', advance='no') '  NDInitFromConfig plan lifecycle           '

   ! Init config
   call shafftConfigNDInit(cfg, ndim, globalShape, SHAFFT_C2C, &
                           hintNda=0, strategy=0, &
                           outputPolicy=SHAFFT_LAYOUT_REDISTRIBUTED, &
                           memLimit=0_c_size_t, comm=MPI_COMM_WORLD, ierr=rc)
   if (rc /= SHAFFT_SUCCESS) goto 992

   ! Create plan and init from config
   call shafftNDCreate(plan, rc)
   if (rc /= SHAFFT_SUCCESS) goto 992

   call shafftNDInitFromConfig(plan, cfg, rc)
   if (rc /= SHAFFT_SUCCESS) goto 992

   ! Verify plan is usable: query alloc size
   call shafftGetAllocSize(plan, allocSize, rc)
   if (rc /= SHAFFT_SUCCESS) goto 992

   if (allocSize <= 0) then
      if (rank == 0) write (*, '(A,I0,A)') 'FAIL (allocSize=', allocSize, ')'
      failed = failed + 1
      call shafftDestroy(plan, rc)
      call shafftConfigNDRelease(cfg)
      return
   end if

   ! Config allocElements should match plan
   if (allocSize /= cfg%allocElements) then
      if (rank == 0) write (*, '(A,I0,A,I0,A)') 'FAIL (plan=', allocSize, &
            ', cfg=', cfg%allocElements, ')'
      failed = failed + 1
      call shafftDestroy(plan, rc)
      call shafftConfigNDRelease(cfg)
      return
   end if

   if (rank == 0) write (*, '(A)') 'PASS'
   passed = passed + 1
   call shafftDestroy(plan, rc)
   call shafftConfigNDRelease(cfg)
   return

992 continue
   failed = failed + 1
   if (rank == 0) then
      call shafftLastErrorMessage(msg)
      write (*, '(A,I0,3A)') 'FAIL (rc=', rc, ', msg=', trim(msg), ')'
   end if
   if (c_associated(plan)) call shafftDestroy(plan, rc)
   if (c_associated(cfg%handle)) call shafftConfigNDRelease(cfg)
end subroutine test_config_nd_init_from_config

!============================================================================
! 1-D Config Tests
!============================================================================

subroutine test_config_1d_lifecycle(passed, failed)
   integer, intent(inout) :: passed, failed
   type(shafft_config_1d_t) :: cfg
   integer(c_size_t) :: globalSize
   integer(c_int) :: rc

   globalSize = 1024
   if (rank == 0) write (*, '(A)', advance='no') '  config1D init/release                    '

   call shafftConfig1DInit(cfg, globalSize, SHAFFT_C2C, MPI_COMM_WORLD, rc)
   if (rc /= SHAFFT_SUCCESS) then
      if (rank == 0) write (*, '(A,I0,A)') 'FAIL (init rc=', rc, ')'
      failed = failed + 1
      return
   end if

   if (.not. c_associated(cfg%handle)) then
      if (rank == 0) write (*, '(A)') 'FAIL (handle null after init)'
      failed = failed + 1
      call shafftConfig1DRelease(cfg)
      return
   end if

   call shafftConfig1DRelease(cfg)

   if (c_associated(cfg%handle)) then
      if (rank == 0) write (*, '(A)') 'FAIL (handle non-null after release)'
      failed = failed + 1
      return
   end if

   if (rank == 0) write (*, '(A)') 'PASS'
   passed = passed + 1
end subroutine test_config_1d_lifecycle

subroutine test_config_1d_fields(passed, failed)
   integer, intent(inout) :: passed, failed
   type(shafft_config_1d_t) :: cfg
   integer(c_size_t) :: globalSize
   integer(c_int) :: rc
   character(len=256) :: msg
   logical :: ok

   globalSize = 512
   if (rank == 0) write (*, '(A)', advance='no') '  config1D resolved fields                 '

   call shafftConfig1DInit(cfg, globalSize, SHAFFT_C2C, MPI_COMM_WORLD, rc)
   if (rc /= SHAFFT_SUCCESS) goto 993

   ok = .true.

   if (cfg%globalSize /= globalSize) then
      if (rank == 0) write (*, '(A,I0,A)') 'FAIL (globalSize=', cfg%globalSize, ')'
      ok = .false.
   end if

   if (cfg%precision_ /= SHAFFT_C2C) then
      if (rank == 0) write (*, '(A,I0,A)') 'FAIL (precision=', cfg%precision_, ')'
      ok = .false.
   end if

   if (cfg%allocElements <= 0) then
      if (rank == 0) write (*, '(A,I0,A)') 'FAIL (allocElements=', cfg%allocElements, ')'
      ok = .false.
   end if

   if (cfg%isActive /= 1 .and. cfg%isActive /= 0) then
      if (rank == 0) write (*, '(A,I0,A)') 'FAIL (isActive=', cfg%isActive, ')'
      ok = .false.
   end if

   if (iand(cfg%flags, SHAFFT_CONFIG_RESOLVED) == 0) then
      if (rank == 0) write (*, '(A,I0,A)') 'FAIL (flags=', cfg%flags, ', missing RESOLVED)'
      ok = .false.
   end if

   if (ok) then
      if (rank == 0) write (*, '(A)') 'PASS'
      passed = passed + 1
   else
      failed = failed + 1
   end if

   call shafftConfig1DRelease(cfg)
   return

993 continue
   failed = failed + 1
   if (rank == 0) then
      call shafftLastErrorMessage(msg)
      write (*, '(A,I0,3A)') 'FAIL (init rc=', rc, ', msg=', trim(msg), ')'
   end if
end subroutine test_config_1d_fields

subroutine test_config_1d_resolve(passed, failed)
   integer, intent(inout) :: passed, failed
   type(shafft_config_1d_t) :: cfg
   integer(c_size_t) :: globalSize
   integer(c_int) :: rc
   character(len=256) :: msg
   integer(c_size_t) :: origAlloc

   globalSize = 256
   if (rank == 0) write (*, '(A)', advance='no') '  config1D resolve                         '

   call shafftConfig1DInit(cfg, globalSize, SHAFFT_C2C, MPI_COMM_WORLD, rc)
   if (rc /= SHAFFT_SUCCESS) goto 994

   origAlloc = cfg%allocElements

   ! Re-resolve should succeed and maintain consistency
   call shafftConfig1DResolve(cfg, rc)
   if (rc /= SHAFFT_SUCCESS) goto 994

   ! allocElements should remain the same after re-resolve with no changes
   if (cfg%allocElements /= origAlloc) then
      if (rank == 0) write (*, '(A,I0,A,I0,A)') 'FAIL (alloc changed: ', origAlloc, &
            ' -> ', cfg%allocElements, ')'
      failed = failed + 1
      call shafftConfig1DRelease(cfg)
      return
   end if

   if (rank == 0) write (*, '(A)') 'PASS'
   passed = passed + 1
   call shafftConfig1DRelease(cfg)
   return

994 continue
   failed = failed + 1
   if (rank == 0) then
      call shafftLastErrorMessage(msg)
      write (*, '(A,I0,3A)') 'FAIL (rc=', rc, ', msg=', trim(msg), ')'
   end if
   if (c_associated(cfg%handle)) call shafftConfig1DRelease(cfg)
end subroutine test_config_1d_resolve

subroutine test_config_1d_init_from_config(passed, failed)
   integer, intent(inout) :: passed, failed
   type(shafft_config_1d_t) :: cfg
   type(c_ptr) :: plan = c_null_ptr
   integer(c_size_t) :: globalSize, allocSize
   integer(c_int) :: rc
   character(len=256) :: msg

   globalSize = 1024
   if (rank == 0) write (*, '(A)', advance='no') '  1DInitFromConfig plan lifecycle           '

   call shafftConfig1DInit(cfg, globalSize, SHAFFT_C2C, MPI_COMM_WORLD, rc)
   if (rc /= SHAFFT_SUCCESS) goto 995

   call shafft1DCreate(plan, rc)
   if (rc /= SHAFFT_SUCCESS) goto 995

   call shafft1DInitFromConfig(plan, cfg, rc)
   if (rc /= SHAFFT_SUCCESS) goto 995

   call shafftGetAllocSize(plan, allocSize, rc)
   if (rc /= SHAFFT_SUCCESS) goto 995

   if (allocSize <= 0) then
      if (rank == 0) write (*, '(A,I0,A)') 'FAIL (allocSize=', allocSize, ')'
      failed = failed + 1
      call shafftDestroy(plan, rc)
      call shafftConfig1DRelease(cfg)
      return
   end if

   if (allocSize /= cfg%allocElements) then
      if (rank == 0) write (*, '(A,I0,A,I0,A)') 'FAIL (plan=', allocSize, &
            ', cfg=', cfg%allocElements, ')'
      failed = failed + 1
      call shafftDestroy(plan, rc)
      call shafftConfig1DRelease(cfg)
      return
   end if

   if (rank == 0) write (*, '(A)') 'PASS'
   passed = passed + 1
   call shafftDestroy(plan, rc)
   call shafftConfig1DRelease(cfg)
   return

995 continue
   failed = failed + 1
   if (rank == 0) then
      call shafftLastErrorMessage(msg)
      write (*, '(A,I0,3A)') 'FAIL (rc=', rc, ', msg=', trim(msg), ')'
   end if
   if (c_associated(plan)) call shafftDestroy(plan, rc)
   if (c_associated(cfg%handle)) call shafftConfig1DRelease(cfg)
end subroutine test_config_1d_init_from_config

!============================================================================
! Communicator Query Tests
!============================================================================

subroutine test_get_communicator_nd(passed, failed)
   integer, intent(inout) :: passed, failed
   type(shafft_config_nd_t) :: cfg
   type(c_ptr) :: plan = c_null_ptr
   integer, parameter :: ndim = 2
   integer(c_size_t) :: globalShape(ndim)
   integer(c_int) :: rc, outComm
   character(len=256) :: msg

   globalShape = [32, 32]
   if (rank == 0) write (*, '(A)', advance='no') '  shafftGetCommunicator (ND)                '

   call shafftConfigNDInit(cfg, ndim, globalShape, SHAFFT_C2C, &
                           hintNda=0, strategy=0, &
                           outputPolicy=SHAFFT_LAYOUT_REDISTRIBUTED, &
                           memLimit=0_c_size_t, comm=MPI_COMM_WORLD, ierr=rc)
   if (rc /= SHAFFT_SUCCESS) goto 996

   call shafftNDCreate(plan, rc)
   if (rc /= SHAFFT_SUCCESS) goto 996

   call shafftNDInitFromConfig(plan, cfg, rc)
   if (rc /= SHAFFT_SUCCESS) goto 996

   call shafftGetCommunicator(plan, outComm, rc)
   if (rc /= SHAFFT_SUCCESS) goto 996

   ! Active ranks should get a valid (non-MPI_COMM_NULL) communicator
   if (cfg%isActive == 1) then
      if (outComm == MPI_COMM_NULL) then
         if (rank == 0) write (*, '(A)') 'FAIL (active rank got MPI_COMM_NULL)'
         failed = failed + 1
         call shafftDestroy(plan, rc)
         call shafftConfigNDRelease(cfg)
         return
      end if
      call MPI_Comm_free(outComm, ierr)
   end if

   if (rank == 0) write (*, '(A)') 'PASS'
   passed = passed + 1
   call shafftDestroy(plan, rc)
   call shafftConfigNDRelease(cfg)
   return

996 continue
   failed = failed + 1
   if (rank == 0) then
      call shafftLastErrorMessage(msg)
      write (*, '(A,I0,3A)') 'FAIL (rc=', rc, ', msg=', trim(msg), ')'
   end if
   if (c_associated(plan)) call shafftDestroy(plan, rc)
   if (c_associated(cfg%handle)) call shafftConfigNDRelease(cfg)
end subroutine test_get_communicator_nd

subroutine test_get_communicator_1d(passed, failed)
   integer, intent(inout) :: passed, failed
   type(shafft_config_1d_t) :: cfg
   type(c_ptr) :: plan = c_null_ptr
   integer(c_size_t) :: globalSize
   integer(c_int) :: rc, outComm
   character(len=256) :: msg

   globalSize = 512
   if (rank == 0) write (*, '(A)', advance='no') '  shafftGetCommunicator (1D)                '

   call shafftConfig1DInit(cfg, globalSize, SHAFFT_C2C, MPI_COMM_WORLD, rc)
   if (rc /= SHAFFT_SUCCESS) goto 997

   call shafft1DCreate(plan, rc)
   if (rc /= SHAFFT_SUCCESS) goto 997

   call shafft1DInitFromConfig(plan, cfg, rc)
   if (rc /= SHAFFT_SUCCESS) goto 997

   call shafftGetCommunicator(plan, outComm, rc)
   if (rc /= SHAFFT_SUCCESS) goto 997

   if (cfg%isActive == 1) then
      if (outComm == MPI_COMM_NULL) then
         if (rank == 0) write (*, '(A)') 'FAIL (active rank got MPI_COMM_NULL)'
         failed = failed + 1
         call shafftDestroy(plan, rc)
         call shafftConfig1DRelease(cfg)
         return
      end if
      call MPI_Comm_free(outComm, ierr)
   end if

   if (rank == 0) write (*, '(A)') 'PASS'
   passed = passed + 1
   call shafftDestroy(plan, rc)
   call shafftConfig1DRelease(cfg)
   return

997 continue
   failed = failed + 1
   if (rank == 0) then
      call shafftLastErrorMessage(msg)
      write (*, '(A,I0,3A)') 'FAIL (rc=', rc, ', msg=', trim(msg), ')'
   end if
   if (c_associated(plan)) call shafftDestroy(plan, rc)
   if (c_associated(cfg%handle)) call shafftConfig1DRelease(cfg)
end subroutine test_get_communicator_1d

end program test_f03_config
