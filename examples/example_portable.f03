!> \example example_portable.f03
! Backend-agnostic Fortran example using the portable buffer API.
program example_portable
   use iso_c_binding
   use mpi_f08
   use shafft

   implicit none

   integer, parameter :: ndim = 3
   integer, parameter :: nda = 1
   integer(c_int) :: dimensions(ndim) = [64, 64, 32]

   type(c_ptr) :: plan
   integer(c_int) :: rank, nprocs, ierr, rc
   integer(c_size_t) :: alloc_size
   integer(c_int) :: subsize(ndim), offset(ndim)
   integer :: i, local_elems
   integer :: major, minor, patch
   character(len=32) :: backend_name, version_str

   ! Buffers
   complex(c_double), pointer :: data_buf(:) => null()
   complex(c_double), pointer :: work_buf(:) => null()
   complex(c_double), allocatable :: host_data(:)
   complex(c_double), allocatable :: result(:)

   ! Initialize MPI
   call MPI_Init(ierr)
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

   ! Print version info on rank 0
   if (rank == 0) then
      call shafftGetVersion(major, minor, patch)
      call shafftGetBackendName(backend_name)
      call shafftGetVersionString(version_str)
      write(*,'(A,A,A,A,A)') 'SHAFFT version ', trim(version_str), ' (backend: ', trim(backend_name), ')'
   end if

   ! Create and initialize plan with NDA decomposition (double precision)
   call shafftPlanNDA(plan, ndim, nda, dimensions, SHAFFT_Z2Z, MPI_COMM_WORLD%MPI_VAL)

   ! Get local layout
   call shafftGetLayout(plan, subsize, offset, SHAFFT_TENSOR_LAYOUT_CURRENT)

   ! Get required allocation size
   call shafftGetAllocSize(plan, alloc_size)

   ! Calculate local elements
   local_elems = 1
   do i = 1, ndim
      local_elems = local_elems * subsize(i)
   end do

   ! Allocate buffers using portable API
   call shafftAllocBuffer(alloc_size, data_buf)
   call shafftAllocBuffer(alloc_size, work_buf)

   ! Initialize host data
   allocate(host_data(alloc_size))
   do i = 1, int(alloc_size)
      host_data(i) = cmplx(real(rank + 1, c_double), 0.5_c_double, kind=c_double)
   end do

   ! Copy to buffer
   call shafftCopyToBuffer(data_buf, host_data, alloc_size)

   ! Set buffers
   call shafftSetBuffers(plan, data_buf, work_buf)

   ! Execute forward FFT
   call shafftExecute(plan, SHAFFT_FORWARD)

   ! Execute backward FFT
   call shafftExecute(plan, SHAFFT_BACKWARD)

   ! Normalize
   call shafftNormalize(plan)

   ! Get buffers (may have been swapped)
   call shafftGetBuffers(plan, alloc_size, data_buf, work_buf)

   ! Copy result back to host
   allocate(result(alloc_size))
   call shafftCopyFromBuffer(result, data_buf, alloc_size)

   ! Print first few values on rank 0
   if (rank == 0) then
      write(*,'(A)') 'Fortran example completed successfully!'
      write(*,'(A)', advance='no') 'Result[1..4] = '
      do i = 1, min(4, int(alloc_size))
         write(*,'(A,F7.4,A,F7.4,A)', advance='no') '(', real(result(i)), ',', aimag(result(i)), ') '
      end do
      write(*,*)
   end if

   ! Cleanup
   deallocate(result)
   deallocate(host_data)
   call shafftDestroy(plan)
   call shafftFreeBuffer(data_buf)
   call shafftFreeBuffer(work_buf)

   call MPI_Finalize(ierr)

end program example_portable
