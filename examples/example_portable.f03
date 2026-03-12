!> @example example_portable.f03
!> Backend-portable SHAFFT Fortran example.
program example_portable
use iso_c_binding
use mpi
use shafft
implicit none

integer :: dims(3), commDims(3), nda, commSize, rc, ierr, rank
integer(c_size_t) :: subsize(3), offset(3), elem_count, localElems
type(c_ptr) :: plan
complex(c_float), pointer :: data_buf(:) => null(), work_buf(:) => null()
complex(c_float), allocatable :: host(:), spectrum(:), result(:)

call MPI_Init(ierr)
call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)

dims = [64, 64, 32]
commDims = [0, 0, 0]

call shafftConfigurationND(dims, SHAFFT_C2C, commDims, nda, subsize, offset, &
                           commSize, SHAFFT_MINIMIZE_NDA, 0_c_size_t, MPI_COMM_WORLD, rc)

call shafftNDCreate(plan, rc)
call shafftNDInit(plan, commDims, dims, SHAFFT_C2C, MPI_COMM_WORLD, SHAFFT_LAYOUT_REDISTRIBUTED, rc)
call shafftGetAllocSize(plan, elem_count, rc)

localElems = subsize(1)*subsize(2)*subsize(3)

call shafftAllocBuffer(elem_count, data_buf, rc)
call shafftAllocBuffer(elem_count, work_buf, rc)

allocate (host(elem_count), spectrum(elem_count), result(elem_count))
host = (0.0, 0.0)
if (rank == 0 .and. localElems > 0) host(1) = (1.0, 0.0)

call shafftCopyToBuffer(data_buf, host, elem_count, rc)
call shafftSetBuffers(plan, data_buf, work_buf, rc)
call shafftPlan(plan, rc)

call shafftExecute(plan, SHAFFT_FORWARD, rc)
call shafftNormalize(plan, rc)
call shafftGetBuffers(plan, elem_count, data_buf, work_buf, rc)
call shafftCopyFromBuffer(spectrum, data_buf, elem_count, rc)

if (rank == 0) then
  print '(A,4(A,F9.6,A,F9.6,A))', 'Spectrum[1..4] = ', &
    '(', real(spectrum(1)), ',', aimag(spectrum(1)), ') ', &
    '(', real(spectrum(2)), ',', aimag(spectrum(2)), ') ', &
    '(', real(spectrum(3)), ',', aimag(spectrum(3)), ') ', &
    '(', real(spectrum(4)), ',', aimag(spectrum(4)), ')'
end if

call shafftExecute(plan, SHAFFT_BACKWARD, rc)
call shafftNormalize(plan, rc)
call shafftGetBuffers(plan, elem_count, data_buf, work_buf, rc)
call shafftCopyFromBuffer(result, data_buf, elem_count, rc)

if (rank == 0) then
  print '(A,4(A,F9.6,A,F9.6,A))', 'Result[1..4]   = ', &
    '(', real(result(1)), ',', aimag(result(1)), ') ', &
    '(', real(result(2)), ',', aimag(result(2)), ') ', &
    '(', real(result(3)), ',', aimag(result(3)), ') ', &
    '(', real(result(4)), ',', aimag(result(4)), ')'
end if

deallocate (host, spectrum, result)
call shafftFreeBuffer(data_buf, rc)
call shafftFreeBuffer(work_buf, rc)
call shafftDestroy(plan, rc)

call MPI_Finalize(ierr)
end program example_portable
