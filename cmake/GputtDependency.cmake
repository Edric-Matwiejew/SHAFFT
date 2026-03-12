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
- ``gputt`` target (IMPORTED library)
- ``GPUTT_PROVIDER`` cache variable: ``"local"``, ``"system"``, ``"cached"``, or ``"fetched"``
- ``GPUTT_VERSION`` variable (if detected)
- ``GPUTT_INCLUDE_DIR`` path to gpuTT include directory

Usage
^^^^^
.. code-block:: cmake

   include(GputtDependency)
   target_link_libraries(myapp PRIVATE gputt)

#]=======================================================================]

include_guard(GLOBAL)

if(CMAKE_VERSION VERSION_LESS 3.25)
  message(FATAL_ERROR
    "GputtDependency: CMake 3.25 or newer is required; found ${CMAKE_VERSION}")
endif()

option(GPUTT_AUTO_FETCH
  "Fetch and build if find_package fails" ON)

set(GPUTT_SOURCE_DIR "" CACHE PATH
  "Path to local gpuTT source directory (overrides fetch)")

set(GPUTT_GIT_TAG "master" CACHE STRING
  "Git tag/branch for gpuTT")

set(GPUTT_GIT_REPO "https://github.com/ROCm/gpuTT.git" CACHE STRING
  "Git repository URL")

option(GPUTT_BUILD_PYTHON
  "Build gpuTT Python bindings" OFF)

function(_gputt_ensure_targets)
  if(TARGET gputt::gputt AND NOT TARGET gputt)
    add_library(gputt ALIAS gputt::gputt)
  elseif(TARGET gputt AND NOT TARGET gputt::gputt)
    add_library(gputt::gputt ALIAS gputt)
  endif()
endfunction()

mark_as_advanced(
  GPUTT_AUTO_FETCH
  GPUTT_GIT_REPO
  GPUTT_GIT_TAG
  GPUTT_BUILD_PYTHON
  GPUTT_PROVIDER
  GPUTT_VERSION
  GPUTT_INCLUDE_DIR
)

if(NOT CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_HIP_ARCHITECTURES)
  message(WARNING
    "GputtDependency: Neither CMAKE_CUDA_ARCHITECTURES nor CMAKE_HIP_ARCHITECTURES is set. "
    "gpuTT requires one of these to be defined for the target GPU.")
endif()

set(GPUTT_ROCM_ROOT "" CACHE PATH
  "ROCm installation root; auto-detected if empty")
mark_as_advanced(GPUTT_ROCM_ROOT)

if(NOT GPUTT_ROCM_ROOT)
  set(_gputt_rocm_guess "")
  if(DEFINED ROCM_PATH AND IS_DIRECTORY "${ROCM_PATH}")
    set(_gputt_rocm_guess "${ROCM_PATH}")
  elseif(DEFINED ENV{ROCM_PATH} AND IS_DIRECTORY "$ENV{ROCM_PATH}")
    set(_gputt_rocm_guess "$ENV{ROCM_PATH}")
  elseif(IS_DIRECTORY "/opt/rocm")
    set(_gputt_rocm_guess "/opt/rocm")
  endif()
  if(_gputt_rocm_guess)
    set(GPUTT_ROCM_ROOT "${_gputt_rocm_guess}" CACHE PATH
      "ROCm installation root; auto-detected if empty")
  endif()
endif()

if(CMAKE_HIP_ARCHITECTURES AND GPUTT_ROCM_ROOT)
  if(NOT DEFINED CACHE{HIP_PATH})
    set(HIP_PATH "${GPUTT_ROCM_ROOT}" CACHE PATH "Path to HIP installation")
  endif()
endif()

if(GPUTT_SOURCE_DIR)
  if(NOT IS_DIRECTORY "${GPUTT_SOURCE_DIR}")
    message(FATAL_ERROR
      "GputtDependency: GPUTT_SOURCE_DIR is set but does not exist: ${GPUTT_SOURCE_DIR}")
  endif()

  message(STATUS "GputtDependency: Using local source at ${GPUTT_SOURCE_DIR}")

  block()
    set(GPUTT_BUILD_PYTHON ${GPUTT_BUILD_PYTHON} CACHE BOOL "" FORCE)
    add_subdirectory("${GPUTT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/_gputt" EXCLUDE_FROM_ALL)
  endblock()

  if(NOT TARGET gputt)
    message(FATAL_ERROR "GputtDependency: Local source did not create 'gputt' target")
  endif()

  get_target_property(_type gputt TYPE)
  if(NOT _type STREQUAL "INTERFACE_LIBRARY")
    set_target_properties(gputt PROPERTIES SYSTEM TRUE)
  endif()

  target_include_directories(gputt PUBLIC
    $<BUILD_INTERFACE:${GPUTT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${GPUTT_SOURCE_DIR}/include/gputt>
  )

  _gputt_ensure_targets()

  set(GPUTT_PROVIDER "local" CACHE STRING "gpuTT provider" FORCE)
  set(GPUTT_VERSION "local" CACHE STRING "gpuTT version" FORCE)
  set(GPUTT_INCLUDE_DIR "${GPUTT_SOURCE_DIR}/include" CACHE PATH "gpuTT include directory" FORCE)

  message(STATUS "GputtDependency: Using local gpuTT from ${GPUTT_SOURCE_DIR}")
  return()
endif()

find_package(gputt CONFIG QUIET)

if(gputt_FOUND)
  message(STATUS "GputtDependency: Found system gpuTT at ${gputt_DIR}")

  _gputt_ensure_targets()

  if(TARGET gputt OR TARGET gputt::gputt)
    set(GPUTT_PROVIDER "system" CACHE STRING "gpuTT provider" FORCE)
    set(GPUTT_VERSION "${gputt_VERSION}" CACHE STRING "gpuTT version" FORCE)
    
    if(TARGET gputt::gputt)
      get_target_property(_inc gputt::gputt INTERFACE_INCLUDE_DIRECTORIES)
    else()
      get_target_property(_inc gputt INTERFACE_INCLUDE_DIRECTORIES)
    endif()
    if(_inc)
      list(GET _inc 0 _first_inc)
      set(GPUTT_INCLUDE_DIR "${_first_inc}" CACHE PATH "gpuTT include directory" FORCE)
    endif()

    message(STATUS "GputtDependency: Using system gpuTT ${gputt_VERSION}")
    return()
  else()
    message(WARNING "GputtDependency: Package found but targets missing; will fetch instead")
  endif()
endif()

if(DEFINED FETCHCONTENT_BASE_DIR)
  set(_gputt_base_dir "${FETCHCONTENT_BASE_DIR}")
else()
  set(_gputt_base_dir "${PROJECT_SOURCE_DIR}/.deps")
endif()

set(_gputt_install_dir "${_gputt_base_dir}/gputt-install")
set(_gputt_cached_config "${_gputt_install_dir}/lib/cmake/gputt/gputtConfig.cmake")

if(EXISTS "${_gputt_cached_config}")
  set(_gputt_cached_valid TRUE)
  foreach(_check_dir "${_gputt_install_dir}/include" "${_gputt_install_dir}/include/gputt")
    if(NOT IS_DIRECTORY "${_check_dir}")
      set(_gputt_cached_valid FALSE)
      message(STATUS "GputtDependency: Cached install missing directory: ${_check_dir}")
    endif()
  endforeach()
  if(NOT EXISTS "${_gputt_install_dir}/lib/libgputt.a")
    set(_gputt_cached_valid FALSE)
    message(STATUS "GputtDependency: Cached install missing library: ${_gputt_install_dir}/lib/libgputt.a")
  endif()

  if(NOT _gputt_cached_valid)
    message(STATUS "GputtDependency: Cached gpuTT installation is incomplete; will rebuild")
    file(REMOVE_RECURSE "${_gputt_base_dir}/gputt-prefix/src/gputt_external-stamp")
    message(STATUS "GputtDependency: Cleared stale ExternalProject stamp files")
  else()
    message(STATUS "GputtDependency: Found previously-built gpuTT in ${_gputt_install_dir}")

    set(gputt_DIR "${_gputt_install_dir}/lib/cmake/gputt")
    find_package(gputt CONFIG QUIET)

    if(gputt_FOUND)
      _gputt_ensure_targets()

      if(TARGET gputt OR TARGET gputt::gputt)
        set(GPUTT_PROVIDER "cached" CACHE STRING "gpuTT provider" FORCE)
        set(GPUTT_VERSION "${gputt_VERSION}" CACHE STRING "gpuTT version" FORCE)
        set(GPUTT_INCLUDE_DIR "${_gputt_install_dir}/include" CACHE PATH "gpuTT include directory" FORCE)

        message(STATUS "GputtDependency: Using cached gpuTT from ${_gputt_install_dir}")
        return()
      endif()
    endif()

    message(STATUS "GputtDependency: Cached gpuTT config invalid; will rebuild")
    file(REMOVE_RECURSE "${_gputt_base_dir}/gputt-prefix/src/gputt_external-stamp")
  endif()
endif()

if(NOT GPUTT_AUTO_FETCH)
  message(FATAL_ERROR
    "GputtDependency: gpuTT not found and GPUTT_AUTO_FETCH is OFF.\n"
    "Options:\n"
    "  - Set gputt_DIR to existing installation\n"
    "  - Set GPUTT_SOURCE_DIR to local source directory\n"
    "  - Set GPUTT_AUTO_FETCH=ON to download automatically")
endif()

include(ExternalProject)

message(STATUS "GputtDependency: Fetching gpuTT ${GPUTT_GIT_TAG}...")

set(_gputt_gpu_flags "")
if(CMAKE_HIP_ARCHITECTURES)
  list(APPEND _gputt_gpu_flags
    "-DCMAKE_HIP_ARCHITECTURES=${CMAKE_HIP_ARCHITECTURES}"
  )
elseif(CMAKE_CUDA_ARCHITECTURES)
  list(APPEND _gputt_gpu_flags
    "-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}"
  )
endif()

set(_gputt_cmake_args
  "-DCMAKE_INSTALL_PREFIX=${_gputt_install_dir}"
  "-DCMAKE_INSTALL_LIBDIR=lib"
  "-DCMAKE_INSTALL_INCLUDEDIR=include"
  "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
  "-DGPUTT_BUILD_PYTHON=${GPUTT_BUILD_PYTHON}"
  ${_gputt_gpu_flags}
)

if(CMAKE_HIP_COMPILER)
  list(APPEND _gputt_cmake_args
    "-DCMAKE_CXX_COMPILER=${CMAKE_HIP_COMPILER}"
    "-DCMAKE_HIP_COMPILER=${CMAKE_HIP_COMPILER}"
  )
  if(CMAKE_HIP_FLAGS)
    list(APPEND _gputt_cmake_args
      "-DCMAKE_HIP_FLAGS=${CMAKE_HIP_FLAGS}"
      "-DCMAKE_CXX_FLAGS=${CMAKE_HIP_FLAGS}"
    )
  endif()
else()
  list(APPEND _gputt_cmake_args
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
  )
endif()

if(GPUTT_ROCM_ROOT)
  list(APPEND _gputt_cmake_args
    "-DHIP_PATH=${GPUTT_ROCM_ROOT}"
    "-DCMAKE_PREFIX_PATH=${GPUTT_ROCM_ROOT}|${GPUTT_ROCM_ROOT}/lib/cmake/hip"
  )
endif()

# Remove stale include and add missing <cstddef> include for C++ builds
set(_gputt_patch_cmd
  ${CMAKE_COMMAND} -E echo "Patching gpuTT sources..."
  COMMAND sed -i.bak "s|include/gputt/gputt_runtime.h||g" <SOURCE_DIR>/CMakeLists.txt
  COMMAND sed -i.bak "s/DESTINATION \\./DESTINATION lib/g" <SOURCE_DIR>/CMakeLists.txt
  COMMAND sed -i.bak "/#include <stdint.h>/a #include <cstddef>" <SOURCE_DIR>/include/gputt/gputt.h
)

ExternalProject_Add(gputt_external
  GIT_REPOSITORY "${GPUTT_GIT_REPO}"
  GIT_TAG "${GPUTT_GIT_TAG}"
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
  PREFIX "${_gputt_base_dir}/gputt-prefix"
  SOURCE_DIR "${_gputt_base_dir}/gputt-src"
  BINARY_DIR "${_gputt_base_dir}/gputt-build"
  INSTALL_DIR "${_gputt_install_dir}"
  PATCH_COMMAND ${_gputt_patch_cmd}
  LIST_SEPARATOR "|"
  CMAKE_ARGS ${_gputt_cmake_args}
  UPDATE_DISCONNECTED TRUE
  BUILD_BYPRODUCTS "${_gputt_install_dir}/lib/libgputt.a"
)

message(STATUS "GputtDependency: Will build and install gpuTT ${GPUTT_GIT_TAG} to ${_gputt_install_dir}")

file(MAKE_DIRECTORY "${_gputt_install_dir}/include")
file(MAKE_DIRECTORY "${_gputt_install_dir}/include/gputt")

add_library(gputt STATIC IMPORTED GLOBAL)
set_target_properties(gputt PROPERTIES
  IMPORTED_LOCATION "${_gputt_install_dir}/lib/libgputt.a"
  INTERFACE_INCLUDE_DIRECTORIES "${_gputt_install_dir}/include;${_gputt_install_dir}/include/gputt"
)
add_dependencies(gputt gputt_external)

_gputt_ensure_targets()

set(GPUTT_PROVIDER "fetched" CACHE STRING "gpuTT provider" FORCE)
set(GPUTT_VERSION "${GPUTT_GIT_TAG}" CACHE STRING "gpuTT version" FORCE)
set(GPUTT_INCLUDE_DIR "${_gputt_install_dir}/include" CACHE PATH "gpuTT include directory" FORCE)

message(STATUS "gpuTT provider: ${GPUTT_PROVIDER}")
message(STATUS "gpuTT version: ${GPUTT_VERSION}")
