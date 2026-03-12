!> @file test_utils.f90
!! @brief Common utilities for SHAFFT Fortran tests
!!
!! Mirrors the C++ test_utils.hpp for consistent validation standards.
module test_utils_f
use iso_c_binding
use mpi

implicit none

!> Base tolerances (before size scaling) - matches C++ test_utils.hpp
real(c_float), parameter :: TOL_F = 1.0e-5
real(c_double), parameter :: TOL_D = 1.0e-12

contains

!> @brief Reduce pass/fail across all MPI ranks.
!! @param localPass .true. if this rank passes, .false. if fails
!! @param comm MPI communicator
!! @return .true. only if ALL ranks pass
function allRanksPassF(localPass, comm) result(globalPass)
   logical, intent(in) :: localPass
   integer, intent(in) :: comm
   logical :: globalPass
   integer :: localInt, globalInt, ierr
   
   localInt = merge(1, 0, localPass)
   call MPI_Allreduce(localInt, globalInt, 1, MPI_INTEGER, MPI_MIN, comm, ierr)
   globalPass = (globalInt == 1)
end function allRanksPassF

!> @brief Compute FFTW-style relative error for single precision.
!!
!! Computes both L∞ and L2 relative errors across all MPI ranks,
!! scaled by sqrt(log N) per FFTW methodology.
!!
!! @param out Output data (result)
!! @param ref Reference data (original)
!! @param local_count Number of local elements
!! @param globalN Total global element count (for scaling)
!! @param comm MPI communicator
!! @param base_tol Base tolerance (before scaling)
!! @return .true. if error is within tolerance, .false. if exceeds
function checkRelErrorSP(out, ref, local_count, globalN, comm, base_tol) result(pass)
   complex(c_float), intent(in) :: out(:), ref(:)
   integer(c_size_t), intent(in) :: local_count, globalN
   integer, intent(in) :: comm
   real(c_double), intent(in) :: base_tol
   logical :: pass
   
   real(c_double) :: localMaxErr, local_max_ref, local_err2, local_ref2
   real(c_double) :: global_max_err, global_max_ref, global_err2, global_ref2
   real(c_double) :: er, ei, err_mag, rr, ri, ref_mag
   real(c_double) :: rel_linf, rel_l2, scale, tol
   integer :: i, ierr
   
   localMaxErr = 0.0d0
   local_max_ref = 0.0d0
   local_err2 = 0.0d0
   local_ref2 = 0.0d0
   
   do i = 1, int(local_count)
      er = real(out(i), c_double) - real(ref(i), c_double)
      ei = aimag(out(i)) - aimag(ref(i))
      err_mag = sqrt(er*er + ei*ei)
      rr = real(ref(i), c_double)
      ri = aimag(ref(i))
      ref_mag = sqrt(rr*rr + ri*ri)
      
      if (err_mag > localMaxErr) localMaxErr = err_mag
      if (ref_mag > local_max_ref) local_max_ref = ref_mag
      local_err2 = local_err2 + err_mag*err_mag
      local_ref2 = local_ref2 + ref_mag*ref_mag
   end do
   
   call MPI_Allreduce(localMaxErr, global_max_err, 1, MPI_DOUBLE_PRECISION, MPI_MAX, comm, ierr)
   call MPI_Allreduce(local_max_ref, global_max_ref, 1, MPI_DOUBLE_PRECISION, MPI_MAX, comm, ierr)
   call MPI_Allreduce(local_err2, global_err2, 1, MPI_DOUBLE_PRECISION, MPI_SUM, comm, ierr)
   call MPI_Allreduce(local_ref2, global_ref2, 1, MPI_DOUBLE_PRECISION, MPI_SUM, comm, ierr)
   
   if (global_max_ref > 0.0d0) then
      rel_linf = global_max_err / global_max_ref
   else
      rel_linf = global_max_err
   end if
   
   if (global_ref2 > 0.0d0) then
      rel_l2 = sqrt(global_err2 / global_ref2)
   else
      rel_l2 = sqrt(global_err2)
   end if
   
   ! Scale tolerance by sqrt(log N) per FFTW methodology
   scale = sqrt(log(max(real(globalN, c_double), 2.0d0)))
   tol = base_tol * scale
   
   pass = (rel_linf <= tol) .and. (rel_l2 <= tol)
end function checkRelErrorSP

!> @brief Compute FFTW-style relative error for double precision.
function check_rel_error_dp(out, ref, local_count, globalN, comm, base_tol) result(pass)
   complex(c_double), intent(in) :: out(:), ref(:)
   integer(c_size_t), intent(in) :: local_count, globalN
   integer, intent(in) :: comm
   real(c_double), intent(in) :: base_tol
   logical :: pass
   
   real(c_double) :: localMaxErr, local_max_ref, local_err2, local_ref2
   real(c_double) :: global_max_err, global_max_ref, global_err2, global_ref2
   real(c_double) :: er, ei, err_mag, ref_mag
   real(c_double) :: rel_linf, rel_l2, scale, tol
   integer :: i, ierr
   
   localMaxErr = 0.0d0
   local_max_ref = 0.0d0
   local_err2 = 0.0d0
   local_ref2 = 0.0d0
   
   do i = 1, int(local_count)
      er = real(out(i)) - real(ref(i))
      ei = aimag(out(i)) - aimag(ref(i))
      err_mag = sqrt(er*er + ei*ei)
      ref_mag = sqrt(real(ref(i))**2 + aimag(ref(i))**2)
      
      if (err_mag > localMaxErr) localMaxErr = err_mag
      if (ref_mag > local_max_ref) local_max_ref = ref_mag
      local_err2 = local_err2 + err_mag*err_mag
      local_ref2 = local_ref2 + ref_mag*ref_mag
   end do
   
   call MPI_Allreduce(localMaxErr, global_max_err, 1, MPI_DOUBLE_PRECISION, MPI_MAX, comm, ierr)
   call MPI_Allreduce(local_max_ref, global_max_ref, 1, MPI_DOUBLE_PRECISION, MPI_MAX, comm, ierr)
   call MPI_Allreduce(local_err2, global_err2, 1, MPI_DOUBLE_PRECISION, MPI_SUM, comm, ierr)
   call MPI_Allreduce(local_ref2, global_ref2, 1, MPI_DOUBLE_PRECISION, MPI_SUM, comm, ierr)
   
   if (global_max_ref > 0.0d0) then
      rel_linf = global_max_err / global_max_ref
   else
      rel_linf = global_max_err
   end if
   
   if (global_ref2 > 0.0d0) then
      rel_l2 = sqrt(global_err2 / global_ref2)
   else
      rel_l2 = sqrt(global_err2)
   end if
   
   scale = sqrt(log(max(real(globalN, c_double), 2.0d0)))
   tol = base_tol * scale
   
   pass = (rel_linf <= tol) .and. (rel_l2 <= tol)
end function check_rel_error_dp

!> @brief Compute product of integer array elements.
function product_int(arr) result(p)
   integer(c_int), intent(in) :: arr(:)
   integer(c_size_t) :: p
   integer :: i
   p = 1
   do i = 1, size(arr)
      p = p * int(arr(i), c_size_t)
   end do
end function product_int

end module test_utils_f
