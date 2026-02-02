!> \example example.f03
! Fortran example using hipfort for explicit GPU memory management.
program example_hip
use iso_c_binding
use mpi
use hipfort
use hipfort_check
use shafft
implicit none

integer :: ierr, rank
integer(c_int), parameter :: ndim = 3
integer(c_int), target :: dims(ndim) = [64_c_int, 64_c_int, 32_c_int]
integer(c_int), target :: subsize(ndim), offset(ndim)
type(c_ptr) :: plan, d_data_p, d_work_p
complex(c_float), pointer :: d_data(:), d_work(:)
complex(c_float), allocatable, target :: h(:)
integer(c_size_t) :: elem_count
integer(c_int) :: local_elems, i

call MPI_Init(ierr)
call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)

! NDA planner
call shafftPlanNDA(plan, ndim, 1_c_int, dims, SHAFFT_C2C, MPI_COMM_WORLD, ierr)

call shafftGetLayout(plan, subsize, offset, SHAFFT_TENSOR_LAYOUT_CURRENT, ierr)
call shafftGetAllocSize(plan, elem_count, ierr)

call hipCheck(hipMalloc(d_data_p, elem_count*c_sizeof((0.0_c_float, 0.0_c_float))))
call hipCheck(hipMalloc(d_work_p, elem_count*c_sizeof((0.0_c_float, 0.0_c_float))))

local_elems = 1
do i = 1, ndim
   local_elems = local_elems*subsize(i)
end do
allocate (h(local_elems))

! Initialize with single global impulse at [0,0,0] (only on rank 0)
h = (0.0_c_float, 0.0_c_float)
if (rank == 0) then
   h(1) = (1.0_c_float, 1.0_c_float)
end if

  call hipCheck(hipMemcpy(d_data_p, c_loc(h(1)), size(h,kind=c_size_t)*c_sizeof(h(1)), hipMemcpyHostToDevice))

call c_f_pointer(d_data_p, d_data, [local_elems])
call c_f_pointer(d_work_p, d_work, [local_elems])

call shafftSetBuffers(plan, d_data, d_work, ierr)
call shafftExecute(plan, SHAFFT_FORWARD, ierr)
call shafftNormalize(plan, ierr)
call shafftGetBuffers(plan, size(d_data, kind=c_size_t), d_data, d_work, ierr)

  call hipCheck(hipMemcpy(c_loc(h(1)), c_loc(d_data(1)), size(h,kind=c_size_t)*c_sizeof(h(1)), hipMemcpyDeviceToHost))

! Print result on rank 0
if (rank == 0) then
   write (*, '(A)', advance='no') 'Result[0..3] = '
   do i = 1, min(4, local_elems)
      write (*, '(A,F8.5,A,F8.5,A)', advance='no') '(', real(h(i)), ',', aimag(h(i)), ') '
   end do
   write (*, *)
end if

call shafftDestroy(plan, ierr)
call hipCheck(hipFree(d_data_p))
call hipCheck(hipFree(d_work_p))
deallocate (h)

call MPI_Finalize(ierr)
end program example_hip
