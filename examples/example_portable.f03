!> \example example_portable.f03
! Backend-agnostic Fortran example using the portable buffer API.
program example_portable
use iso_c_binding
use mpi
use shafft

implicit none

integer, parameter :: ndim = 3
integer, parameter :: nda = 1
integer(c_int) :: dimensions(ndim) = [64, 64, 32]

type(c_ptr) :: plan
integer(c_int) :: rank, nprocs, ierr
integer(c_size_t) :: alloc_size
integer(c_int) :: subsize(ndim), offset(ndim)
integer :: i, local_elems

! Buffers (single precision to match C/C++ examples)
complex(c_float), pointer :: data_buf(:) => null()
complex(c_float), pointer :: work_buf(:) => null()
complex(c_float), allocatable :: host_data(:)
complex(c_float), allocatable :: result(:)

! Initialize MPI
call MPI_Init(ierr)
call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

! Create and initialize plan with NDA decomposition (single precision)
call shafftPlanNDA(plan, ndim, nda, dimensions, SHAFFT_C2C, MPI_COMM_WORLD, ierr)

! Get local layout
call shafftGetLayout(plan, subsize, offset, SHAFFT_TENSOR_LAYOUT_CURRENT, ierr)

! Get required allocation size
call shafftGetAllocSize(plan, alloc_size, ierr)

! Calculate local elements
local_elems = 1
do i = 1, ndim
   local_elems = local_elems*subsize(i)
end do

! Allocate buffers using portable API
call shafftAllocBuffer(alloc_size, data_buf, ierr)
call shafftAllocBuffer(alloc_size, work_buf, ierr)

! Initialize host data with single global impulse at [0,0,0] (only on rank 0)
allocate (host_data(alloc_size))
host_data = cmplx(0.0_c_float, 0.0_c_float, kind=c_float)
if (rank == 0 .and. alloc_size > 0) then
   host_data(1) = cmplx(1.0_c_float, 1.0_c_float, kind=c_float)
end if

! Copy to buffer
call shafftCopyToBuffer(data_buf, host_data, alloc_size, ierr)

! Set buffers
call shafftSetBuffers(plan, data_buf, work_buf, ierr)

! Execute forward FFT
call shafftExecute(plan, SHAFFT_FORWARD, ierr)

! Normalize
call shafftNormalize(plan, ierr)

! Get buffers (may have been swapped)
call shafftGetBuffers(plan, alloc_size, data_buf, work_buf, ierr)

! Copy result back to host
allocate (result(alloc_size))
call shafftCopyFromBuffer(result, data_buf, alloc_size, ierr)

! Print first few values on rank 0
if (rank == 0) then
   write (*, '(A)', advance='no') 'Result[0..3] = '
   do i = 1, min(4, int(alloc_size))
      write (*, '(A,F8.5,A,F8.5,A)', advance='no') '(', real(result(i)), ',', aimag(result(i)), ') '
   end do
   write (*, *)
end if

! Cleanup
deallocate (result)
deallocate (host_data)
call shafftDestroy(plan, ierr)
call shafftFreeBuffer(data_buf, ierr)
call shafftFreeBuffer(work_buf, ierr)

call MPI_Finalize(ierr)

end program example_portable
