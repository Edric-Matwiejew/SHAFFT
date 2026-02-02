cmake_minimum_required(VERSION 3.21)
project(SHAFFT_Tests LANGUAGES C CXX)

if(SHAFFT_BUILD_FORTRAN)
  enable_language(Fortran)
endif()

if(SHAFFT_BUILD_FORTRAN)
  find_package(MPI REQUIRED COMPONENTS C CXX Fortran)
else()
  find_package(MPI REQUIRED COMPONENTS C CXX)
endif()

if(TARGET shafftc++)
  message(STATUS "Using shafftc++ target from parent build")
  set(
    SHAFFT_INCLUDE_DIR
    ${CMAKE_BINARY_DIR}/include # generated shafft_config.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
  )
else()
  set(SHAFFT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
  if(EXISTS ${SHAFFT_ROOT}/include/shafft.hpp)
    set(SHAFFT_INCLUDE_DIR ${SHAFFT_ROOT}/include)
    set(SHAFFT_LIB_HINT ${SHAFFT_ROOT}/lib ${SHAFFT_ROOT}/lib64)
  else()
    find_path(SHAFFT_INCLUDE_DIR shafft.hpp HINTS /usr/local/include $ENV{SHAFFT_ROOT}/include)
    set(
      SHAFFT_LIB_HINT
      /usr/local/lib
      /usr/local/lib64
      $ENV{SHAFFT_ROOT}/lib
      $ENV{SHAFFT_ROOT}/lib64
    )
  endif()

  file(READ ${SHAFFT_INCLUDE_DIR}/shafft_config.h SHAFFT_CONFIG_CONTENT)
  string(REGEX MATCH "#define SHAFFT_BACKEND_HIPFFT 1" HIP_ENABLED ${SHAFFT_CONFIG_CONTENT})
  if(HIP_ENABLED)
    list(APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH} /opt/rocm)
    find_package(hip QUIET)
    if(hip_FOUND)
      set(HIP_INCLUDE_DIRS ${hip_INCLUDE_DIRS})
    else()
      set(HIP_INCLUDE_DIRS $ENV{ROCM_PATH}/include /opt/rocm/include)
    endif()
    add_compile_definitions(__HIP_PLATFORM_AMD__)
  else()
    set(HIP_INCLUDE_DIRS "")
  endif()

  find_library(SHAFFT_CXX_LIB NAMES shafftc++ HINTS ${SHAFFT_LIB_HINT})
  if(NOT SHAFFT_CXX_LIB)
    message(FATAL_ERROR "Could not find shafftc++ library (set SHAFFT_ROOT or install SHAFFT).")
  endif()
  add_library(shafftc++ SHARED IMPORTED)
  set_target_properties(
    shafftc++
    PROPERTIES
      IMPORTED_LOCATION ${SHAFFT_CXX_LIB}
      INTERFACE_INCLUDE_DIRECTORIES
        "$<BUILD_INTERFACE:${SHAFFT_INCLUDE_DIR};${HIP_INCLUDE_DIRS}>;$<INSTALL_INTERFACE:include>"
  )

  find_library(SHAFFT_C_LIB NAMES shafftc HINTS ${SHAFFT_LIB_HINT})
  if(NOT SHAFFT_C_LIB)
    message(FATAL_ERROR "Could not find shafftc library required for C API tests.")
  endif()
  add_library(shafftc SHARED IMPORTED)
  set_target_properties(
    shafftc
    PROPERTIES
      IMPORTED_LOCATION ${SHAFFT_C_LIB}
      INTERFACE_INCLUDE_DIRECTORIES
        "$<BUILD_INTERFACE:${SHAFFT_INCLUDE_DIR};${HIP_INCLUDE_DIRS}>;$<INSTALL_INTERFACE:include>"
  )

  if(SHAFFT_BUILD_FORTRAN)
    find_library(SHAFFT_F03_LIB NAMES shafftf03 HINTS ${SHAFFT_LIB_HINT})
    if(NOT SHAFFT_F03_LIB)
      message(FATAL_ERROR "SHAFFT_BUILD_FORTRAN=ON but shafftf03 library not found.")
    endif()
    add_library(shafftf03 SHARED IMPORTED)
    set_target_properties(
      shafftf03
      PROPERTIES
        IMPORTED_LOCATION ${SHAFFT_F03_LIB}
        INTERFACE_INCLUDE_DIRECTORIES
          "$<BUILD_INTERFACE:${SHAFFT_INCLUDE_DIR};${HIP_INCLUDE_DIRS}>;$<INSTALL_INTERFACE:include>"
    )
  endif()
endif()

# Ensure HIP_ENABLED is available in both parent and standalone builds.
# Prefer the top-level cache variables if present; fall back to config header.
if(NOT DEFINED HIP_ENABLED)
  if(DEFINED SHAFFT_BACKEND_HIPFFT)
    set(HIP_ENABLED ${SHAFFT_BACKEND_HIPFFT})
  elseif(DEFINED SHAFFT_ENABLE_HIPFFT)
    set(HIP_ENABLED ${SHAFFT_ENABLE_HIPFFT})
  elseif(EXISTS ${CMAKE_BINARY_DIR}/include/shafft/shafft_config.h)
    file(READ ${CMAKE_BINARY_DIR}/include/shafft/shafft_config.h SHAFFT_CONFIG_CONTENT)
    string(REGEX MATCH "#define SHAFFT_BACKEND_HIPFFT 1" HIP_ENABLED ${SHAFFT_CONFIG_CONTENT})
  elseif(EXISTS ${SHAFFT_INCLUDE_DIR}/shafft_config.h)
    file(READ ${SHAFFT_INCLUDE_DIR}/shafft_config.h SHAFFT_CONFIG_CONTENT)
    string(REGEX MATCH "#define SHAFFT_BACKEND_HIPFFT 1" HIP_ENABLED ${SHAFFT_CONFIG_CONTENT})
  else()
    set(HIP_ENABLED 0)
  endif()
endif()

# Common include directories
set(TEST_INCLUDES ${SHAFFT_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)

function(add_mpi_test TEST_NAME TEST_SOURCE)
  add_executable(${TEST_NAME} ${TEST_SOURCE})
  target_include_directories(${TEST_NAME} PRIVATE ${TEST_INCLUDES})
  target_link_libraries(${TEST_NAME} PRIVATE shafftc++ MPI::MPI_CXX)

  # Parse optional RANKS argument (default: 1 2 4)
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
if(HIP_ENABLED)
  add_unit_test(test_hipfft_method unit/test_hipfft_method.cpp)
  target_include_directories(
    test_hipfft_method
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpp/fft/hipfft_method
      ${gputt_SOURCE_DIR}/include/gputt
  )
endif()
add_unit_test(test_fft_highdim unit/test_fft_highdim.cpp)

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(test_fft_highdim PRIVATE OpenMP::OpenMP_CXX)
endif()

# Only find FFTW if not already found by parent project
if(NOT TARGET FFTW::Double)
  find_package(FFTW QUIET COMPONENTS DOUBLE_LIB)
  if(NOT FFTW_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      FindFFTW
      GIT_REPOSITORY https://github.com/egpbos/findfftw.git
      GIT_TAG master
      GIT_SHALLOW TRUE
      UPDATE_DISCONNECTED TRUE
    )
    FetchContent_MakeAvailable(FindFFTW)
    list(APPEND CMAKE_MODULE_PATH "${findfftw_SOURCE_DIR}")
    find_package(FFTW QUIET COMPONENTS DOUBLE_LIB)
  endif()
endif()
if(HIP_ENABLED)
  add_unit_test(test_fft_partial_hipfft unit/test_fft_partial.cpp)
  target_include_directories(
    test_fft_partial_hipfft
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpp/fft
      ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpp/fft/hipfft_method
      ${gputt_SOURCE_DIR}/include/gputt
  )
endif()

# Component tests: verify third-party library integration (gpuTT, hipFFT)
if(HIP_ENABLED)
  add_executable(test_transpose_fft component/test_transpose_fft.cpp)
  target_include_directories(
    test_transpose_fft
    PRIVATE ${TEST_INCLUDES} ${gputt_SOURCE_DIR}/include/gputt
  )
  target_link_libraries(test_transpose_fft PRIVATE shafftc++ MPI::MPI_CXX hip::hipfft gputt)
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
)
if(HIP_ENABLED)
  list(APPEND CHECK_DEPS test_hipfft_method)
endif()
if(HIP_ENABLED)
  list(APPEND CHECK_DEPS test_fft_partial_hipfft)
endif()
if(SHAFFT_BACKEND_FFTW)
  list(APPEND CHECK_DEPS test_fft_partial_fftw)
endif()
if(SHAFFT_BUILD_FORTRAN)
  list(APPEND CHECK_DEPS test_f03_roundtrip test_f03_api)
endif()

add_mpi_test(test_roundtrip integration/cpp/test_roundtrip.cpp RANKS 1 2 4)
add_mpi_test(test_delta integration/cpp/test_delta.cpp RANKS 1 2 4)
add_mpi_test(test_correctness integration/cpp/test_correctness.cpp RANKS 1 2 4)
add_mpi_test(test_highdim integration/cpp/test_highdim.cpp RANKS 1 2 4)
add_mpi_test(test_errors integration/cpp/test_errors.cpp RANKS 1 2)
add_mpi_test(test_layout integration/cpp/test_layout.cpp RANKS 1 2 4 8)
add_mpi_test(test_axes integration/cpp/test_axes.cpp RANKS 1 2 4)
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

add_mpi_c_test(test_c_roundtrip ${CMAKE_CURRENT_SOURCE_DIR}/integration/c/test_c_roundtrip.c RANKS 1 2 4)
add_mpi_c_test(test_c_api ${CMAKE_CURRENT_SOURCE_DIR}/integration/c/test_c_api.c RANKS 1 2 4)

if(SHAFFT_BUILD_FORTRAN)
  function(add_mpi_fortran_test TEST_NAME TEST_SOURCE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_include_directories(
      ${TEST_NAME}
      PRIVATE
        ${TEST_INCLUDES}
        ${CMAKE_BINARY_DIR}/include # For shafft.mod
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

  add_mpi_fortran_test(test_f03_roundtrip 
        ${CMAKE_CURRENT_SOURCE_DIR}/integration/fortran/test_f03_roundtrip.f03 
        RANKS 1 2 4
  )

  add_mpi_fortran_test(test_f03_api 
        ${CMAKE_CURRENT_SOURCE_DIR}/integration/fortran/test_f03_api.f03 
        RANKS 1 2 4
  )
endif()

add_custom_target(
  check
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
  DEPENDS ${CHECK_DEPS}
  COMMENT "Running SHAFFT test suite"
)
