!> \example example.f03
! Fortran example using hipfort for explicit GPU memory management.
program example_hip
use iso_c_binding
use mpi
use hipfort
use hipfort_check
use shafft
implicit none

integer :: ierr, rank, nprocs
integer(c_int), parameter :: ndim = 3
integer(c_int), target :: dims(ndim) = [64_c_int, 64_c_int, 32_c_int]
integer(c_int), target :: commDims(ndim)
integer(c_int) :: nda, commSize
integer(c_size_t), target :: subsize(ndim), offset(ndim)
type(c_ptr) :: plan, d_data_p, d_work_p
complex(c_float), pointer :: d_data(:), d_work(:)
complex(c_float), allocatable, target :: h(:)
integer(c_size_t) :: elem_count, localElems
integer(c_int) :: e, i

call MPI_Init(ierr)
call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

! Step 1: Get configuration
commDims = 0
nda = 0
call shafftConfigurationND(ndim, dims, SHAFFT_C2C, commDims, nda, subsize, offset, &
                           commSize, SHAFFT_MINIMIZE_NDA, 0_c_size_t, MPI_COMM_WORLD, e)
if (e /= 0) then
  write (*, '(A,I0)') 'shafftConfigurationND failed: ', e
  call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
end if

! Step 2: Create and initialize plan
call shafftNDInit(plan, commDims, dims, SHAFFT_C2C, MPI_COMM_WORLD, SHAFFT_LAYOUT_REDISTRIBUTED, e)
if (e /= 0) then
  write (*, '(A,I0)') 'shafftNDInit failed: ', e
  call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
end if

call shafftGetAllocSize(plan, elem_count, e)

localElems = subsize(1)*subsize(2)*subsize(3)

call hipCheck(hipMalloc(d_data_p, elem_count*c_sizeof((0.0_c_float, 0.0_c_float))))
call hipCheck(hipMalloc(d_work_p, elem_count*c_sizeof((0.0_c_float, 0.0_c_float))))

allocate (h(elem_count))

! Initialize with single global impulse at [0,0,0] (only on rank 0)
h = (0.0_c_float, 0.0_c_float)
if (rank == 0 .and. localElems > 0) then
  h(1) = (1.0_c_float, 1.0_c_float)
end if

call hipCheck(hipMemcpy(d_data_p, c_loc(h(1)), elem_count*c_sizeof(h(1)), hipMemcpyHostToDevice))

call c_f_pointer(d_data_p, d_data, [elem_count])
call c_f_pointer(d_work_p, d_work, [elem_count])

call shafftSetBuffers(plan, d_data, d_work, e)
call shafftExecute(plan, SHAFFT_FORWARD, e)
call shafftNormalize(plan, e)
call shafftGetBuffers(plan, elem_count, d_data, d_work, e)

call hipCheck(hipMemcpy(c_loc(h(1)), c_loc(d_data(1)), localElems*c_sizeof(h(1)), &
                        hipMemcpyDeviceToHost))

! Print result on rank 0
if (rank == 0) then
  write (*, '(A)', advance='no') 'Result[0..3] = '
  do i = 1, min(4, int(localElems))
    write (*, '(A,F8.5,A,F8.5,A)', advance='no') '(', real(h(i)), ',', aimag(h(i)), ') '
  end do
  write (*, *)
end if

call shafftDestroy(plan, e)
call hipCheck(hipFree(d_data_p))
call hipCheck(hipFree(d_work_p))
deallocate (h)

call MPI_Finalize(ierr)
end program example_hip
