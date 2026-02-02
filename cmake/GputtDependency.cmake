#[=======================================================================[.rst:
GputtDependency
---------------

Finds gpuTT via its CMake config, or optionally fetches and builds it.
gpuTT is a GPU tensor transpose library supporting both CUDA and HIP backends.

Requirements
^^^^^^^^^^^^
- CMake 3.25 or later
- HIP or CUDA installation
- CMAKE_HIP_ARCHITECTURES or CMAKE_CUDA_ARCHITECTURES must be set

Provides
^^^^^^^^
- ``gputt`` target (or ``gputt::gputt`` alias if using system package)
- ``GPUTT_PROVIDER`` cache variable: ``"system"``, ``"local"``, or ``"fetched"``
- ``GPUTT_VERSION`` variable (if detected)

Usage
^^^^^
.. code-block:: cmake

   include(GputtDependency)
   target_link_libraries(myapp PRIVATE gputt)

#]=======================================================================]

include_guard(GLOBAL)

cmake_minimum_required(VERSION 3.25...4.0)

#=============================================================================
# Configuration options
#=============================================================================

option(GPUTT_AUTO_FETCH "Fetch and build if find_package fails" ON)

set(GPUTT_SOURCE_DIR "" CACHE PATH "Path to local gpuTT source directory")

set(GPUTT_GIT_TAG "master" CACHE STRING "Git tag/branch for gpuTT")

set(GPUTT_GIT_REPO "https://github.com/ROCm/gpuTT.git" CACHE STRING "Git repository URL")

option(GPUTT_BUILD_TESTS "Build gpuTT tests" OFF)

option(GPUTT_BUILD_BENCHMARKS "Build gpuTT benchmarks" OFF)

option(GPUTT_BUILD_PYTHON "Build gpuTT Python bindings" OFF)

mark_as_advanced(GPUTT_GIT_REPO GPUTT_GIT_TAG)

#=============================================================================
# Validate prerequisites
#=============================================================================

# Check for GPU architectures
if(NOT CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_HIP_ARCHITECTURES)
  message(
    WARNING
    "GputtDependency: Neither CMAKE_CUDA_ARCHITECTURES nor CMAKE_HIP_ARCHITECTURES is set. "
    "gpuTT requires one of these to be defined for the target GPU."
  )
endif()

#=============================================================================
# Helper: Set HIP_PATH for gpuTT's find_package(hip)
#=============================================================================
# gpuTT uses find_package(hip) with CMAKE_MODULE_PATH set to ${HIP_PATH}/cmake.
# In ROCm 6.x, the hip cmake config is at /opt/rocm/lib/cmake/hip/, but gpuTT
# expects it at ${HIP_PATH}/cmake. We help by setting HIP_PATH appropriately.

function(_gputt_setup_hip_path)
  # Only needed for HIP builds
  if(NOT CMAKE_HIP_ARCHITECTURES)
    return()
  endif()

  # Check if HIP_PATH is already set correctly
  if(DEFINED ENV{HIP_PATH} OR DEFINED CACHE{HIP_PATH})
    return()
  endif()

  # Try to find ROCm installation
  if(DEFINED ENV{ROCM_PATH})
    set(_rocm_path "$ENV{ROCM_PATH}")
  elseif(IS_DIRECTORY "/opt/rocm")
    set(_rocm_path "/opt/rocm")
  else()
    return()
  endif()

  # Set HIP_PATH to ROCm root (gpuTT will look in ${HIP_PATH}/cmake and ${HIP_PATH}/hip/cmake)
  # Also add the modern cmake path to CMAKE_PREFIX_PATH
  set(HIP_PATH "${_rocm_path}" CACHE PATH "Path to HIP installation" FORCE)
  list(APPEND CMAKE_PREFIX_PATH "${_rocm_path}" "${_rocm_path}/lib/cmake/hip")
  set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" PARENT_SCOPE)

  message(VERBOSE "GputtDependency: Set HIP_PATH=${_rocm_path} for gpuTT")
endfunction()

_gputt_setup_hip_path()

#=============================================================================
# Try local source directory first
#=============================================================================

if(GPUTT_SOURCE_DIR)
  if(NOT IS_DIRECTORY "${GPUTT_SOURCE_DIR}")
    message(
      FATAL_ERROR
      "GputtDependency: GPUTT_SOURCE_DIR is set but does not exist: ${GPUTT_SOURCE_DIR}"
    )
  endif()

  message(STATUS "GputtDependency: Using local source at ${GPUTT_SOURCE_DIR}")

  block()
    set(GPUTT_BUILD_TESTS ${GPUTT_BUILD_TESTS} CACHE BOOL "" FORCE)
    set(GPUTT_BUILD_BENCHMARKS ${GPUTT_BUILD_BENCHMARKS} CACHE BOOL "" FORCE)
    set(GPUTT_BUILD_PYTHON ${GPUTT_BUILD_PYTHON} CACHE BOOL "" FORCE)

    add_subdirectory("${GPUTT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/_gputt" EXCLUDE_FROM_ALL)
  endblock()

  if(NOT TARGET gputt)
    message(FATAL_ERROR "GputtDependency: Local source did not create 'gputt' target")
  endif()

  # Mark as SYSTEM to suppress warnings from third-party code
  get_target_property(_type gputt TYPE)
  if(NOT _type STREQUAL "INTERFACE_LIBRARY")
    set_target_properties(gputt PROPERTIES SYSTEM TRUE)
  endif()

  # gpuTT uses PRIVATE includes; add PUBLIC interface so consumers can use headers
  target_include_directories(
    gputt
    PUBLIC
      $<BUILD_INTERFACE:${GPUTT_SOURCE_DIR}/include>
      $<BUILD_INTERFACE:${GPUTT_SOURCE_DIR}/include/gputt>
  )

  set(GPUTT_PROVIDER "local" CACHE INTERNAL "")
  set(GPUTT_VERSION "local" CACHE INTERNAL "")
  set(GPUTT_INCLUDE_DIR "${GPUTT_SOURCE_DIR}/include" CACHE INTERNAL "gpuTT include directory")

  message(STATUS "GputtDependency: Using local gpuTT from ${GPUTT_SOURCE_DIR}")
  return()
endif()

#=============================================================================
# Try system gpuTT
#=============================================================================

find_package(gputt CONFIG QUIET)

if(gputt_FOUND)
  message(VERBOSE "GputtDependency: Found system gpuTT at ${gputt_DIR}")

  # The installed package exports gputt-targets, creating gputt target
  if(NOT TARGET gputt AND TARGET gputt::gputt)
    # Create expected target name as alias
    add_library(gputt ALIAS gputt::gputt)
  endif()

  if(TARGET gputt OR TARGET gputt::gputt)
    set(GPUTT_PROVIDER "system" CACHE INTERNAL "")
    set(GPUTT_VERSION "${gputt_VERSION}" CACHE INTERNAL "")

    message(STATUS "GputtDependency: Using system gpuTT ${gputt_VERSION}")
    return()
  else()
    message(WARNING "GputtDependency: Package found but targets missing; will fetch instead")
  endif()
endif()

#=============================================================================
# Fetch and build gpuTT
#=============================================================================

if(NOT GPUTT_AUTO_FETCH)
  message(
    FATAL_ERROR
    "GputtDependency: gpuTT not found and GPUTT_AUTO_FETCH is OFF.\n"
    "Options:\n"
    "  - Set gputt_DIR to existing installation\n"
    "  - Set GPUTT_SOURCE_DIR to local source directory\n"
    "  - Set GPUTT_AUTO_FETCH=ON to download automatically"
  )
endif()

include(FetchContent)

message(STATUS "GputtDependency: Fetching gpuTT ${GPUTT_GIT_TAG}...")

FetchContent_Declare(
  gputt
  GIT_REPOSITORY "${GPUTT_GIT_REPO}"
  GIT_TAG "${GPUTT_GIT_TAG}"
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
  UPDATE_DISCONNECTED TRUE
)

FetchContent_GetProperties(gputt)

if(NOT gputt_POPULATED)
  FetchContent_Populate(gputt)

  block()
    set(GPUTT_BUILD_TESTS ${GPUTT_BUILD_TESTS} CACHE BOOL "" FORCE)
    set(GPUTT_BUILD_BENCHMARKS ${GPUTT_BUILD_BENCHMARKS} CACHE BOOL "" FORCE)
    set(GPUTT_BUILD_PYTHON ${GPUTT_BUILD_PYTHON} CACHE BOOL "" FORCE)

    add_subdirectory("${gputt_SOURCE_DIR}" "${gputt_BINARY_DIR}" EXCLUDE_FROM_ALL)
  endblock()
endif()

if(NOT TARGET gputt)
  message(
    FATAL_ERROR
    "GputtDependency: Build failed - gputt target not created.\n"
    "Check the build output above for errors."
  )
endif()

# Mark as SYSTEM to suppress warnings from third-party code
get_target_property(_type gputt TYPE)
if(NOT _type STREQUAL "INTERFACE_LIBRARY")
  set_target_properties(gputt PROPERTIES SYSTEM TRUE)
endif()

# gpuTT uses PRIVATE includes; add PUBLIC interface so consumers can use headers
target_include_directories(
  gputt
  PUBLIC
    $<BUILD_INTERFACE:${gputt_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${gputt_SOURCE_DIR}/include/gputt>
)

set(GPUTT_PROVIDER "fetched" CACHE INTERNAL "")
set(GPUTT_VERSION "${GPUTT_GIT_TAG}" CACHE INTERNAL "")
set(GPUTT_INCLUDE_DIR "${gputt_SOURCE_DIR}/include" CACHE INTERNAL "gpuTT include directory")

message(STATUS "GputtDependency: Built gpuTT ${GPUTT_GIT_TAG} from source")
