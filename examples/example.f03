!> \example example.f03
! Fortran example using hipfort for explicit GPU memory management.
program example_hip
  use iso_c_binding
  use mpi
  use hipfort
  use shafft
  implicit none

  integer :: ierr
  integer(c_int), parameter :: ndim = 3
  integer(c_int), target :: dims(ndim) = [64_c_int, 64_c_int, 32_c_int]
  integer(c_int), target :: subsize(ndim), offset(ndim)
  type(c_ptr) :: plan, d_data_p, d_work_p
  complex(c_float), pointer :: d_data(:), d_work(:)
  complex(c_float), allocatable :: h(:)
  integer(c_size_t) :: elem_count
  integer(c_int) :: local_elems, i

  call MPI_Init(ierr)

  ! NDA planner
  call shafftPlanNDA(plan, ndim, 1_c_int, dims, SHAFFT_C2C, MPI_COMM_WORLD)

  ! --- Alternative: Cartesian planner ---
  ! integer(c_int), target :: COMM_DIMS(ndim) = [0_c_int, 0_c_int, 0_c_int]
  ! integer(c_int) :: COMM_SIZE
  ! integer(c_size_t) :: mem_limit = 0_c_size_t
  ! call shafftConfigurationCart(ndim, dims, subsize, offset, COMM_DIMS, COMM_SIZE, &
  !                              SHAFFT_C2C, mem_limit, MPI_COMM_WORLD)
  ! call shafftPlanCart(plan, COMM_DIMS, dims, SHAFFT_C2C, MPI_COMM_WORLD)

  call shafftGetLayout(plan, subsize, offset, SHAFFT_TENSOR_LAYOUT_CURRENT)
  call shafftGetAllocSize(plan, elem_count)

  call hipMalloc(d_data_p, elem_count * c_sizeof((0.0_c_float,0.0_c_float)))
  call hipMalloc(d_work_p, elem_count * c_sizeof((0.0_c_float,0.0_c_float)))

  local_elems = 1
  do i = 1, ndim
     local_elems = local_elems * subsize(i)
  end do
  allocate(h(local_elems))
  h = (0.0_c_float, 0.0_c_float)
  h(1) = (1.0_c_float, 1.0_c_float)

  call hipMemcpy(d_data_p, c_loc(h(1)), size(h,kind=c_size_t)*c_sizeof(h(1)), hipMemcpyHostToDevice)

  call c_f_pointer(d_data_p, d_data, [local_elems])
  call c_f_pointer(d_work_p, d_work, [local_elems])

  call shafftSetBuffers(d_data, d_work)
  call shafftExecute(plan, SHAFFT_FORWARD)
  call shafftNormalize(plan)
  call shafftGetBuffers(d_data, d_work)

  call hipMemcpy(c_loc(h(1)), c_loc(d_data(1)), size(h,kind=c_size_t)*c_sizeof(h(1)), hipMemcpyDeviceToHost)

  call shafftDestroy(plan)
  call hipFree(d_data_p)
  call hipFree(d_work_p)
  deallocate(h)

  call MPI_Finalize(ierr)
end program example_hip
