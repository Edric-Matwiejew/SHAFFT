# SHAFFT Test Suite
# Uses CTest for MPI-based testing

cmake_minimum_required(VERSION 3.21)
project(SHAFFT_Tests LANGUAGES C CXX)

# Enable Fortran if requested
if(SHAFFT_BUILD_FORTRAN)
    enable_language(Fortran)
endif()

# Find MPI (include Fortran component if building Fortran tests)
if(SHAFFT_BUILD_FORTRAN)
    find_package(MPI REQUIRED COMPONENTS C CXX Fortran)
else()
    find_package(MPI REQUIRED COMPONENTS C CXX)
endif()

# Check if we're being built as part of the main SHAFFT build or standalone
if(TARGET shafftc++)
    # Already have the target from parent build
    message(STATUS "Using shafftc++ target from parent build")
    # When building from parent, use parent source directory
    set(SHAFFT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../include)
else()
    # Standalone build - need to find/create the target
    
    # Locate SHAFFT library (either installed or from parent build)
    # First try to find in parent directory (development build)
    set(SHAFFT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
    if(EXISTS ${SHAFFT_ROOT}/include/shafft.hpp)
        set(SHAFFT_INCLUDE_DIR ${SHAFFT_ROOT}/include)
        set(SHAFFT_LIB_DIR ${SHAFFT_ROOT}/lib)
    else()
        # Fallback to installed location
        find_path(SHAFFT_INCLUDE_DIR shafft.hpp
            HINTS /usr/local/include $ENV{SHAFFT_ROOT}/include)
        find_library(SHAFFT_LIB_DIR NAMES shafftc++
            HINTS /usr/local/lib $ENV{SHAFFT_ROOT}/lib)
    endif()

    # Check if SHAFFT was built with HIP backend (check config header)
    file(READ ${SHAFFT_INCLUDE_DIR}/shafft_config.h SHAFFT_CONFIG_CONTENT)
    string(REGEX MATCH "#define SHAFFT_BACKEND_HIPFFT 1" HIP_ENABLED ${SHAFFT_CONFIG_CONTENT})
    if(HIP_ENABLED)
        # Find HIP for header includes
        list(APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH} /opt/rocm)
        find_package(hip QUIET)
        if(hip_FOUND)
            set(HIP_INCLUDE_DIRS ${hip_INCLUDE_DIRS})
        else()
            # Fallback to ROCM_PATH environment
            set(HIP_INCLUDE_DIRS $ENV{ROCM_PATH}/include /opt/rocm/include)
        endif()
        add_compile_definitions(__HIP_PLATFORM_AMD__)
    endif()

    # Create imported target for SHAFFT C++ library
    add_library(shafftc++ SHARED IMPORTED)
    set_target_properties(shafftc++ PROPERTIES
        IMPORTED_LOCATION ${SHAFFT_LIB_DIR}/libshafftc++.so
        INTERFACE_INCLUDE_DIRECTORIES "${SHAFFT_INCLUDE_DIR};${HIP_INCLUDE_DIRS}"
    )
endif()

# Common include directories
set(TEST_INCLUDES
    ${SHAFFT_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Helper function to add an MPI test with multiple rank counts
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
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${TEST_NAME}> ${MPIEXEC_POSTFLAGS}
        )
        set_tests_properties(${TEST_NAME}_${np}ranks PROPERTIES
            TIMEOUT 120
            PROCESSORS ${np}
        )
    endforeach()
endfunction()

# Helper function for unit tests (single rank only, no MPI communication)
function(add_unit_test TEST_NAME TEST_SOURCE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_include_directories(${TEST_NAME} PRIVATE ${TEST_INCLUDES})
    target_link_libraries(${TEST_NAME} PRIVATE shafftc++ MPI::MPI_CXX)
    
    add_test(
        NAME ${TEST_NAME}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${TEST_NAME}> ${MPIEXEC_POSTFLAGS}
    )
    set_tests_properties(${TEST_NAME} PROPERTIES
        TIMEOUT 30
        PROCESSORS 1
        LABELS "unit"
    )
endfunction()

# ==============================================================================
# Unit Tests (no MPI communication, single rank)
# ==============================================================================

add_unit_test(test_error_api unit/test_error_api.cpp)
add_unit_test(test_library_info unit/test_library_info.cpp)
add_unit_test(test_memory unit/test_memory.cpp)

# ==============================================================================
# Integration Tests (require MPI, multiple rank counts)
# ==============================================================================

# Round-trip test: forward + backward + normalize recovers original
add_mpi_test(test_roundtrip integration/cpp/test_roundtrip.cpp RANKS 1 2 4)

# Correctness tests: verify FFT produces expected output
add_mpi_test(test_delta integration/cpp/test_delta.cpp RANKS 1 2 4)

# Mathematical correctness: delta, plane wave, Parseval's theorem
add_mpi_test(test_correctness integration/cpp/test_correctness.cpp RANKS 1 2 4)

# High-dimensional roundtrip: 5D, 7D, 9D, 11D tests
add_mpi_test(test_highdim integration/cpp/test_highdim.cpp RANKS 1 2 4)

# Error handling: invalid inputs return correct error codes
add_mpi_test(test_errors integration/cpp/test_errors.cpp RANKS 1 2)

# Layout tests: verify getLayout returns correct subsize/offset
add_mpi_test(test_layout integration/cpp/test_layout.cpp RANKS 1 2 4 8)

# Axes tests: verify getAxes returns correct ca/da
add_mpi_test(test_axes integration/cpp/test_axes.cpp RANKS 1 2 4)

# Buffer tracking: verify getBuffers tracks swaps after execute
add_mpi_test(test_getbuffers integration/cpp/test_getbuffers.cpp RANKS 1 2 4)

# Cartesian init: test initCart with explicit process grid
add_mpi_test(test_initcart integration/cpp/test_initcart.cpp RANKS 1 2 4)

# Configuration helpers: test configurationNDA and configurationCart
add_mpi_test(test_configuration integration/cpp/test_configuration.cpp RANKS 1 2 4)

# Double precision test
add_mpi_test(test_double integration/cpp/test_double.cpp RANKS 1 2)

# Backend-specific tests: FFTW threading, HIP streams
add_mpi_test(test_backend integration/cpp/test_backend.cpp RANKS 1 2 4)

# Inactive ranks: test graceful handling of excluded MPI ranks
add_mpi_test(test_inactive_ranks integration/cpp/test_inactive_ranks.cpp RANKS 5 6 8)

# Configuration modes: comprehensive tests for configurationNDA and configurationCart
add_mpi_test(test_configuration_modes integration/cpp/test_configuration_modes.cpp RANKS 1 2 4 5 8)

# ==============================================================================
# C API Integration Tests
# ==============================================================================

# Helper function for C tests (links against C library)
function(add_mpi_c_test TEST_NAME TEST_SOURCE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_include_directories(${TEST_NAME} PRIVATE ${TEST_INCLUDES})
    # Link with C library (shafftc) and MPI
    target_link_libraries(${TEST_NAME} PRIVATE shafftc MPI::MPI_CXX)
    # Force C++ compilation for .c files (to handle C++ runtime linking)
    set_source_files_properties(${TEST_SOURCE} PROPERTIES LANGUAGE CXX)
    
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
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${TEST_NAME}> ${MPIEXEC_POSTFLAGS}
        )
        set_tests_properties(${TEST_NAME}_${np}ranks PROPERTIES
            TIMEOUT 120
            PROCESSORS ${np}
        )
    endforeach()
endfunction()

# C API roundtrip test
add_mpi_c_test(test_c_roundtrip ${CMAKE_CURRENT_SOURCE_DIR}/integration/c/test_c_roundtrip.c RANKS 1 2 4)

# C API coverage test
add_mpi_c_test(test_c_api ${CMAKE_CURRENT_SOURCE_DIR}/integration/c/test_c_api.c RANKS 1 2 4)

# ==============================================================================
# Fortran API Integration Tests
# ==============================================================================

if(SHAFFT_BUILD_FORTRAN)
    # Helper function for Fortran tests
    function(add_mpi_fortran_test TEST_NAME TEST_SOURCE)
        add_executable(${TEST_NAME} ${TEST_SOURCE})
        target_include_directories(${TEST_NAME} PRIVATE 
            ${TEST_INCLUDES}
            ${CMAKE_BINARY_DIR}/include  # For shafft.mod
        )
        # Link with Fortran library and MPI
        target_link_libraries(${TEST_NAME} PRIVATE shafftf03 MPI::MPI_Fortran)
        
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
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${TEST_NAME}> ${MPIEXEC_POSTFLAGS}
            )
            set_tests_properties(${TEST_NAME}_${np}ranks PROPERTIES
                TIMEOUT 120
                PROCESSORS ${np}
            )
        endforeach()
    endfunction()

    # Fortran roundtrip test
    add_mpi_fortran_test(test_f03_roundtrip 
        ${CMAKE_CURRENT_SOURCE_DIR}/integration/fortran/test_f03_roundtrip.f03 
        RANKS 1 2 4)

    # Fortran API coverage test
    add_mpi_fortran_test(test_f03_api 
        ${CMAKE_CURRENT_SOURCE_DIR}/integration/fortran/test_f03_api.f03 
        RANKS 1 2 4)
endif()

# Print test summary target
add_custom_target(check
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS test_roundtrip test_delta test_errors test_layout test_double
            test_axes test_getbuffers test_initcart test_configuration
            test_c_roundtrip test_c_api
    COMMENT "Running SHAFFT test suite"
)
