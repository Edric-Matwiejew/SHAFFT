cmake_minimum_required(VERSION 3.21)
project(test_craygtl LANGUAGES C CXX)

# Add parent cmake directory to module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/..")

# Print environment info
message(STATUS "=== Cray GTL Test ===")
message(STATUS "CRAY_CPU_TARGET: $ENV{CRAY_CPU_TARGET}")
message(STATUS "CRAY_MPICH_DIR: $ENV{CRAY_MPICH_DIR}")
message(STATUS "CMAKE_HIP_ARCHITECTURES: ${CMAKE_HIP_ARCHITECTURES}")
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

# Print GTL environment variables if present
foreach(_arch IN ITEMS "amd_gfx90a" "amd_gfx908" "amd_gfx942" "nvidia80" "nvidia90")
  if(DEFINED ENV{PE_MPICH_GTL_DIR_${_arch}})
    message(STATUS "PE_MPICH_GTL_DIR_${_arch}: $ENV{PE_MPICH_GTL_DIR_${_arch}}")
    message(STATUS "PE_MPICH_GTL_LIBS_${_arch}: $ENV{PE_MPICH_GTL_LIBS_${_arch}}")
  endif()
endforeach()

# Try to find CrayGTL
find_package(CrayGTL)

if(CrayGTL_FOUND)
  message(STATUS "")
  message(STATUS "[OK] CrayGTL FOUND")
  message(STATUS "  CrayGTL_LIBRARIES: ${CrayGTL_LIBRARIES}")
  message(STATUS "  CrayGTL_TYPE: ${CrayGTL_TYPE}")
  message(STATUS "  CRAY_GTL_LIBRARY: ${CRAY_GTL_LIBRARY}")
else()
  message(STATUS "")
  message(STATUS "[FAIL] CrayGTL NOT FOUND")
endif()

message(STATUS "=== Test Complete ===")
