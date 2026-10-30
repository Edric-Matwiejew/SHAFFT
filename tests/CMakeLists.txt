cmake_minimum_required(VERSION 3.28)
project(SHAFFT_Tests LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()

if(NOT TARGET shafftc++)
  find_package(shafft CONFIG REQUIRED HINTS "${SHAFFT_ROOT}" "${SHAFFT_PATH}" "$ENV{SHAFFT_ROOT}")
  foreach(_binding IN ITEMS shafftc++ shafftc shafftf03)
    if(TARGET shafft::${_binding})
      add_library(${_binding} ALIAS shafft::${_binding})
    endif()
  endforeach()
endif()

if(SHAFFT_BUILD_FORTRAN)
  enable_language(Fortran)
  find_package(MPI REQUIRED COMPONENTS C CXX Fortran)
else()
  find_package(MPI REQUIRED COMPONENTS C CXX)
endif()

set(HIP_ENABLED ${SHAFFT_BACKEND_HIPFFT})
set(SHAFFT_ENABLE_FFTW ${SHAFFT_BACKEND_FFTW})
if(HIP_ENABLED)
  if(NOT TARGET hip::hipfft)
    find_package(hipfft CONFIG REQUIRED)
  endif()
  if(NOT TARGET gputt::gputt)
    if(GPUTT_SOURCE_DIR)
      set(CMAKE_HIP_PLATFORM "${SHAFFT_GPU_PLATFORM}")
      set(CMAKE_HIP_ARCHITECTURES "${SHAFFT_GPU_ARCHITECTURES}")
      include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/GputtDependency.cmake)
    else()
      find_package(gputt CONFIG REQUIRED)
    endif()
  endif()
endif()

set(TEST_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include)

function(add_mpi_test TEST_NAME TEST_SOURCE)
  add_executable(${TEST_NAME} ${TEST_SOURCE})
  target_include_directories(${TEST_NAME} PRIVATE ${TEST_INCLUDES})
  target_link_libraries(${TEST_NAME} PRIVATE shafftc++ MPI::MPI_CXX)

  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs RANKS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT ARG_RANKS)
    set(ARG_RANKS 1 2 4)
  endif()

  foreach(np ${ARG_RANKS})
    add_test(
      NAME ${TEST_NAME}_${np}ranks
      COMMAND
        ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:${TEST_NAME}> ${MPIEXEC_POSTFLAGS}
    )
    set_tests_properties(${TEST_NAME}_${np}ranks PROPERTIES TIMEOUT 120 PROCESSORS ${np})
  endforeach()
endfunction()

function(add_unit_test TEST_NAME TEST_SOURCE)
  add_executable(${TEST_NAME} ${TEST_SOURCE})
  target_include_directories(${TEST_NAME} PRIVATE ${TEST_INCLUDES})
  target_link_libraries(${TEST_NAME} PRIVATE shafftc++ MPI::MPI_CXX)

  add_test(
    NAME ${TEST_NAME}
    COMMAND
      ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:${TEST_NAME}> ${MPIEXEC_POSTFLAGS}
  )
  set_tests_properties(
    ${TEST_NAME}
    PROPERTIES TIMEOUT 120 PROCESSORS 1 LABELS "unit"
  )
endfunction()

add_unit_test(test_error_api unit/test_error_api.cpp)
add_unit_test(test_library_info unit/test_library_info.cpp)
add_unit_test(test_memory unit/test_memory.cpp)

add_unit_test(test_configuration_nd unit/test_configuration_nd.cpp)
add_unit_test(test_FFT1D unit/test_FFT1D.cpp)

# Config object unit tests
add_unit_test(test_config_nd unit/test_config_nd.cpp)
add_unit_test(test_config_1d unit/test_config_1d.cpp)

# Add multi-rank tests for FFT1D layout correctness (critical for verifying localN vs allocSize)
foreach(np 2 3 4)
  add_test(
    NAME test_FFT1D_${np}ranks
    COMMAND
      ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:test_FFT1D> ${MPIEXEC_POSTFLAGS}
  )
  set_tests_properties(test_FFT1D_${np}ranks PROPERTIES TIMEOUT 120 PROCESSORS ${np} LABELS "unit")
endforeach()

if(HIP_ENABLED)
  add_unit_test(test_hipfft_method unit/test_hipfft_method.cpp)
  target_include_directories(
    test_hipfft_method
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpp/fft/hipfft_method
  )
  target_link_libraries(test_hipfft_method PRIVATE hip::hipfft gputt::gputt)

  add_unit_test(test_superbatch_threshold unit/test_superbatch_threshold.cpp)
  target_include_directories(
    test_superbatch_threshold
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpp/fft/hipfft_method
  )
  target_link_libraries(test_superbatch_threshold PRIVATE hip::hipfft gputt::gputt)
endif()
add_unit_test(test_fft_highdim unit/test_fft_highdim.cpp)

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(test_fft_highdim PRIVATE OpenMP::OpenMP_CXX)
endif()

if(HIP_ENABLED)
  add_unit_test(test_fft_partial_hipfft unit/test_fft_partial.cpp)
  target_include_directories(
    test_fft_partial_hipfft
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpp/fft
      ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpp/fft/hipfft_method
  )
  target_link_libraries(test_fft_partial_hipfft PRIVATE hip::hipfft gputt::gputt)
endif()

# Component tests: verify third-party library integration (gpuTT, hipFFT)
if(HIP_ENABLED)
  add_executable(test_transpose_fft component/test_transpose_fft.cpp)
  target_include_directories(
    test_transpose_fft
    PRIVATE ${TEST_INCLUDES}
  )
  target_link_libraries(test_transpose_fft PRIVATE shafftc++ MPI::MPI_CXX hip::hipfft gputt::gputt)
  add_test(
    NAME test_transpose_fft
    COMMAND
      ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:test_transpose_fft> ${MPIEXEC_POSTFLAGS}
  )
  set_tests_properties(
    test_transpose_fft
    PROPERTIES TIMEOUT 120 PROCESSORS 1 LABELS "component"
  )
endif()
if(SHAFFT_BACKEND_FFTW)
  add_unit_test(test_fft_partial_fftw unit/test_fft_partial_fftw.cpp)
  target_include_directories(
    test_fft_partial_fftw
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpp/fft
      ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpp/fft/fftw_method
  )
endif()

set(
  CHECK_DEPS
  test_error_api
  test_library_info
  test_memory
  test_configuration_nd
  test_fft_highdim
  test_roundtrip
  test_delta
  test_correctness
  test_highdim
  test_errors
  test_layout
  test_axes
  test_getbuffers
  test_initcart
  test_double
  test_backend
  test_inactive_ranks
  test_configuration_modes
  test_c_roundtrip
  test_c_api
  test_c_FFT1D
)
if(HIP_ENABLED)
  list(APPEND CHECK_DEPS test_hipfft_method test_superbatch_threshold)
endif()
if(HIP_ENABLED)
  list(APPEND CHECK_DEPS test_fft_partial_hipfft)
endif()
if(SHAFFT_ENABLE_FFTW)
  list(APPEND CHECK_DEPS test_fft_partial_fftw)
endif()
# FFT1D tests for all backends
list(APPEND CHECK_DEPS test_FFT1D test_FFT1D_roundtrip)
if(SHAFFT_BUILD_FORTRAN)
  list(APPEND CHECK_DEPS test_f03_roundtrip test_f03_api test_f03_config)
endif()

add_mpi_test(test_roundtrip integration/cpp/test_roundtrip.cpp RANKS 1 2 4)
add_mpi_test(test_delta integration/cpp/test_delta.cpp RANKS 1 2 4)
add_mpi_test(test_correctness integration/cpp/test_correctness.cpp RANKS 1 2 4)
add_mpi_test(test_highdim integration/cpp/test_highdim.cpp RANKS 1 2 4)
add_mpi_test(test_errors integration/cpp/test_errors.cpp RANKS 1 2)
add_mpi_test(test_layout integration/cpp/test_layout.cpp RANKS 1 2 4 8)
add_mpi_test(test_axes integration/cpp/test_axes.cpp RANKS 1 2 4)

# FFT1D integration tests (all backends)
add_mpi_test(test_FFT1D_roundtrip integration/cpp/test_FFT1D_roundtrip.cpp RANKS 1 2 4)
add_mpi_test(test_getbuffers integration/cpp/test_getbuffers.cpp RANKS 1 2 4)
add_mpi_test(test_initcart integration/cpp/test_initcart.cpp RANKS 1 2 4)
add_mpi_test(test_double integration/cpp/test_double.cpp RANKS 1 2)
add_mpi_test(test_backend integration/cpp/test_backend.cpp RANKS 1 2 4)
add_mpi_test(test_inactive_ranks integration/cpp/test_inactive_ranks.cpp RANKS 5 6 8)
add_mpi_test(test_configuration_modes integration/cpp/test_configuration_modes.cpp RANKS 1 2 4 5 8)

function(add_mpi_c_test TEST_NAME TEST_SOURCE)
  add_executable(${TEST_NAME} ${TEST_SOURCE})
  target_include_directories(${TEST_NAME} PRIVATE ${TEST_INCLUDES})
  target_link_libraries(${TEST_NAME} PRIVATE shafftc MPI::MPI_CXX)
  set_source_files_properties(${TEST_SOURCE} PROPERTIES LANGUAGE CXX)

  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs RANKS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT ARG_RANKS)
    set(ARG_RANKS 1 2 4)
  endif()

  foreach(np ${ARG_RANKS})
    add_test(
      NAME ${TEST_NAME}_${np}ranks
      COMMAND
        ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:${TEST_NAME}> ${MPIEXEC_POSTFLAGS}
    )
    set_tests_properties(${TEST_NAME}_${np}ranks PROPERTIES TIMEOUT 120 PROCESSORS ${np})
  endforeach()
endfunction()

add_mpi_c_test(
  test_c_roundtrip ${CMAKE_CURRENT_SOURCE_DIR}/integration/c/test_c_roundtrip.c RANKS 1 2 4
)
add_mpi_c_test(test_c_api ${CMAKE_CURRENT_SOURCE_DIR}/integration/c/test_c_api.c RANKS 1 2 4)
add_mpi_c_test(test_c_FFT1D ${CMAKE_CURRENT_SOURCE_DIR}/integration/c/test_c_FFT1D.c RANKS 1 2 4)

if(SHAFFT_BUILD_FORTRAN)
  # Compile test_utils module for Fortran tests
  set(TEST_UTILS_F90 ${CMAKE_CURRENT_SOURCE_DIR}/include/test_utils.f90)

  function(add_mpi_fortran_test TEST_NAME TEST_SOURCE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_include_directories(
      ${TEST_NAME}
      PRIVATE
        ${TEST_INCLUDES}
        ${CMAKE_CURRENT_BINARY_DIR} # For test_utils_f.mod
    )

    target_link_libraries(${TEST_NAME} PRIVATE shafftf03 MPI::MPI_Fortran)

    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs RANKS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT ARG_RANKS)
      set(ARG_RANKS 1 2 4)
    endif()

    foreach(np ${ARG_RANKS})
      add_test(
        NAME ${TEST_NAME}_${np}ranks
        COMMAND
          ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS}
          $<TARGET_FILE:${TEST_NAME}> ${MPIEXEC_POSTFLAGS}
      )
      set_tests_properties(${TEST_NAME}_${np}ranks PROPERTIES TIMEOUT 120 PROCESSORS ${np})
    endforeach()
  endfunction()

  add_library(test_utils_f STATIC ${TEST_UTILS_F90})
  target_link_libraries(test_utils_f PRIVATE MPI::MPI_Fortran shafftf03)
  set_target_properties(
    test_utils_f PROPERTIES Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  if(DEFINED ENV{CRAY_CPU_TARGET})
    target_compile_options(
      test_utils_f PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-fallow-argument-mismatch>
    )
  endif()

  add_mpi_fortran_test(test_f03_roundtrip 
        ${CMAKE_CURRENT_SOURCE_DIR}/integration/fortran/test_f03_roundtrip.f03 
        RANKS 1 2 4
  )
  target_link_libraries(test_f03_roundtrip PRIVATE test_utils_f)

  add_mpi_fortran_test(test_f03_api 
        ${CMAKE_CURRENT_SOURCE_DIR}/integration/fortran/test_f03_api.f03 
        RANKS 1 2 4
  )

  add_mpi_fortran_test(test_f03_config
        ${CMAKE_CURRENT_SOURCE_DIR}/integration/fortran/test_f03_config.f03
        RANKS 1 2 4
  )
endif()

add_custom_target(
  check
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
  DEPENDS ${CHECK_DEPS}
  COMMENT "Running SHAFFT test suite"
)
