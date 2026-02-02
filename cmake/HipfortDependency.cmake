#[=======================================================================[.rst:
HipfortDependency
-----------------

Finds hipfort via its CMake config, or optionally fetches and builds it
using the current Fortran toolchain.

Requirements
^^^^^^^^^^^^
- CMake 3.25 or later
- Fortran language enabled in the parent project
- ROCm installation (unless ``HIPFORT_REQUIRE_ROCM`` is OFF)

Provides
^^^^^^^^
- ``hipfort::hipfort`` target (alias to ``hipfort::hip`` if needed)
- ``HIPFORT_PROVIDER`` cache variable: ``"system"`` or ``"fetched"``
- ``HIPFORT_VERSION`` variable (if detected)

Usage
^^^^^
.. code-block:: cmake

   include(HipfortDependency)
   target_link_libraries(myapp PRIVATE hipfort::hipfort)

#]=======================================================================]

include_guard(GLOBAL)

cmake_minimum_required(VERSION 3.25...4.0)

#=============================================================================
# Configuration options
#=============================================================================

option(HIPFORT_AUTO_FETCH "Fetch and build if find_package fails" ON)

option(HIPFORT_VERIFY_MOD_COMPAT "Verify .mod compatibility with Fortran compiler" ON)

option(HIPFORT_REQUIRE_ROCM "Require ROCm installation" ON)

set(HIPFORT_GIT_TAG "" CACHE STRING "Git tag/branch; auto-detected from ROCm if empty")

set(HIPFORT_GIT_REPO "https://github.com/ROCm/hipfort.git" CACHE STRING "Git repository URL")

set(HIPFORT_SOURCE_URL "" CACHE STRING "Source URL or path; overrides git")

set(HIPFORT_SOURCE_URL_HASH "" CACHE STRING "URL hash (e.g. SHA256=...)")

mark_as_advanced(HIPFORT_GIT_REPO HIPFORT_SOURCE_URL HIPFORT_SOURCE_URL_HASH)

#=============================================================================
# Validate prerequisites
#=============================================================================

if(NOT CMAKE_Fortran_COMPILER)
  message(
    FATAL_ERROR
    "HipfortDependency: Fortran compiler not found. "
    "Call project(... LANGUAGES Fortran) before including this module."
  )
endif()

if(NOT CMAKE_C_COMPILER_LOADED AND NOT CMAKE_CXX_COMPILER_LOADED)
  enable_language(C)
endif()

#=============================================================================
# ROCm detection
#=============================================================================

set(HIPFORT_ROCM_ROOT "" CACHE PATH "ROCm installation root; auto-detected if empty")
mark_as_advanced(HIPFORT_ROCM_ROOT)

if(NOT HIPFORT_ROCM_ROOT)
  block(PROPAGATE HIPFORT_ROCM_ROOT)
    if(DEFINED ROCM_PATH AND IS_DIRECTORY "${ROCM_PATH}")
      set(HIPFORT_ROCM_ROOT "${ROCM_PATH}" CACHE PATH "" FORCE)
    elseif(NOT HIPFORT_ROCM_ROOT)
      foreach(_path IN LISTS CMAKE_PREFIX_PATH)
        if(_path MATCHES "[Rr][Oo][Cc][Mm]" AND IS_DIRECTORY "${_path}")
          set(HIPFORT_ROCM_ROOT "${_path}" CACHE PATH "" FORCE)
          break()
        endif()
      endforeach()
    endif()
    if(NOT HIPFORT_ROCM_ROOT AND IS_DIRECTORY "/opt/rocm")
      set(HIPFORT_ROCM_ROOT "/opt/rocm" CACHE PATH "" FORCE)
    endif()
  endblock()
endif()

if(HIPFORT_REQUIRE_ROCM AND NOT HIPFORT_ROCM_ROOT)
  message(
    FATAL_ERROR
    "HipfortDependency: ROCm not found. Options:\n"
    "  - Set ROCM_PATH to your ROCm installation\n"
    "  - Add ROCm to CMAKE_PREFIX_PATH\n"
    "  - Set HIPFORT_REQUIRE_ROCM=OFF if ROCm is not needed"
  )
endif()

if(HIPFORT_ROCM_ROOT)
  list(PREPEND CMAKE_PREFIX_PATH "${HIPFORT_ROCM_ROOT}" "${HIPFORT_ROCM_ROOT}/lib/cmake/hipfort")
  message(VERBOSE "HipfortDependency: ROCm root = ${HIPFORT_ROCM_ROOT}")
endif()

#=============================================================================
# ROCm version detection (for auto-tagging)
#=============================================================================

set(HIPFORT_ROCM_VERSION "" CACHE STRING "Detected ROCm version" FORCE)
mark_as_advanced(HIPFORT_ROCM_VERSION)

if(HIPFORT_ROCM_ROOT)
  if(EXISTS "${HIPFORT_ROCM_ROOT}/.info/version")
    file(READ "${HIPFORT_ROCM_ROOT}/.info/version" _ver_content)
    if(_ver_content MATCHES "([0-9]+\\.[0-9]+\\.[0-9]+)")
      set(HIPFORT_ROCM_VERSION "${CMAKE_MATCH_1}" CACHE STRING "" FORCE)
    endif()
  endif()

  if(NOT HIPFORT_ROCM_VERSION)
    set(_hip_version_file "${HIPFORT_ROCM_ROOT}/lib/cmake/hip/hip-config-version.cmake")
    if(EXISTS "${_hip_version_file}")
      file(READ "${_hip_version_file}" _ver_content)
      if(_ver_content MATCHES "([0-9]+\\.[0-9]+\\.[0-9]+)")
        set(HIPFORT_ROCM_VERSION "${CMAKE_MATCH_1}" CACHE STRING "" FORCE)
      endif()
    endif()
  endif()

  if(HIPFORT_ROCM_VERSION)
    message(VERBOSE "HipfortDependency: ROCm version = ${HIPFORT_ROCM_VERSION}")
  endif()
endif()

if(NOT HIPFORT_GIT_TAG)
  if(HIPFORT_ROCM_VERSION)
    set(
      HIPFORT_GIT_TAG
      "rocm-${HIPFORT_ROCM_VERSION}"
      CACHE STRING
      "Git tag/branch; auto-detected from ROCm if empty"
      FORCE
    )
  else()
    set(
      HIPFORT_GIT_TAG
      "rocm-6.2.0"
      CACHE STRING
      "Git tag/branch; auto-detected from ROCm if empty"
      FORCE
    )
    if(HIPFORT_REQUIRE_ROCM)
      message(WARNING "HipfortDependency: Could not detect ROCm version, using ${HIPFORT_GIT_TAG}")
    endif()
  endif()
endif()

#=============================================================================
# Try system hipfort
#=============================================================================

# Request 'hip' component - required for hipfort::hip target
find_package(hipfort CONFIG QUIET COMPONENTS hip)

if(hipfort_FOUND)
  message(VERBOSE "HipfortDependency: Found system hipfort at ${hipfort_DIR}")

  if(TARGET hipfort::hip)
    set(_hipfort_target hipfort::hip)
  elseif(TARGET hipfort::hipfort)
    set(_hipfort_target hipfort::hipfort)
  else()
    message(WARNING "HipfortDependency: Package found but targets missing; will fetch instead")
    set(hipfort_FOUND FALSE)
  endif()
endif()

if(hipfort_FOUND AND HIPFORT_VERIFY_MOD_COMPAT)
  # Use the platform-specific target for include dirs - it has the .mod file paths
  # hipfort builds either amdgcn (AMD) or nvptx (NVIDIA) depending on GPU platform
  if(TARGET hipfort::hipfort-amdgcn)
    get_target_property(_inc_dirs hipfort::hipfort-amdgcn INTERFACE_INCLUDE_DIRECTORIES)
  elseif(TARGET hipfort::hipfort-nvptx)
    get_target_property(_inc_dirs hipfort::hipfort-nvptx INTERFACE_INCLUDE_DIRECTORIES)
  else()
    get_target_property(_inc_dirs ${_hipfort_target} INTERFACE_INCLUDE_DIRECTORIES)
  endif()

  set(_mod_flags "")
  foreach(_dir IN LISTS _inc_dirs)
    if(NOT _dir MATCHES "^\\$<")
      string(APPEND _mod_flags " -I${_dir}")
    endif()
  endforeach()

  try_compile(
    _mod_compat_ok
    SOURCE_FROM_CONTENT _hipfort_check.f90 "program p; use hipfort; end program"
    CMAKE_FLAGS
      "-DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}"
      "-DCMAKE_Fortran_FLAGS=${CMAKE_Fortran_FLAGS}${_mod_flags}"
    OUTPUT_VARIABLE _mod_compat_log
  )

  if(_mod_compat_ok)
    message(VERBOSE "HipfortDependency: Module compatibility check passed")
  else()
    message(
      STATUS
      "HipfortDependency: System hipfort .mod files incompatible with "
      "${CMAKE_Fortran_COMPILER_ID} ${CMAKE_Fortran_COMPILER_VERSION} "
      "(likely built with different compiler); will fetch and build instead"
    )
    message(VERBOSE "Module compatibility check output:\n${_mod_compat_log}")
    set(hipfort_FOUND FALSE)
  endif()
endif()

if(hipfort_FOUND)
  if(NOT TARGET hipfort::hipfort AND TARGET hipfort::hip)
    add_library(hipfort::hipfort ALIAS hipfort::hip)
  endif()

  set(HIPFORT_PROVIDER "system" CACHE INTERNAL "")
  set(HIPFORT_VERSION "${hipfort_VERSION}" CACHE INTERNAL "")

  message(STATUS "HipfortDependency: Using system hipfort ${hipfort_VERSION}")
  return()
endif()

#=============================================================================
# Fetch and build hipfort
#=============================================================================

if(NOT HIPFORT_AUTO_FETCH)
  message(
    FATAL_ERROR
    "HipfortDependency: hipfort not found and HIPFORT_AUTO_FETCH is OFF.\n"
    "Options:\n"
    "  - Set hipfort_DIR to existing installation\n"
    "  - Add hipfort location to CMAKE_PREFIX_PATH\n"
    "  - Set HIPFORT_AUTO_FETCH=ON to download automatically"
  )
endif()

include(FetchContent)

message(STATUS "HipfortDependency: Fetching hipfort ${HIPFORT_GIT_TAG}...")

if(HIPFORT_SOURCE_URL)
  if(NOT HIPFORT_SOURCE_URL MATCHES "^[a-zA-Z][a-zA-Z0-9+.-]*://")
    cmake_path(ABSOLUTE_PATH HIPFORT_SOURCE_URL OUTPUT_VARIABLE _source_url)
    if(NOT EXISTS "${_source_url}")
      message(FATAL_ERROR "HipfortDependency: Source path not found: ${_source_url}")
    endif()
  else()
    set(_source_url "${HIPFORT_SOURCE_URL}")
  endif()

  set(_fc_args URL "${_source_url}" DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  if(HIPFORT_SOURCE_URL_HASH)
    list(APPEND _fc_args URL_HASH "${HIPFORT_SOURCE_URL_HASH}")
  endif()
else()
  set(
    _fc_args
    GIT_REPOSITORY
    "${HIPFORT_GIT_REPO}"
    GIT_TAG
    "${HIPFORT_GIT_TAG}"
    GIT_SHALLOW
    TRUE
    GIT_PROGRESS
    TRUE
  )
endif()

FetchContent_Declare(hipfort ${_fc_args})
FetchContent_GetProperties(hipfort)

if(NOT hipfort_POPULATED)
  FetchContent_Populate(hipfort)

  # Patch: hipfort uses CMAKE_SOURCE_DIR which breaks as a subdirectory
  set(_cmakelists "${hipfort_SOURCE_DIR}/CMakeLists.txt")
  if(EXISTS "${_cmakelists}")
    file(READ "${_cmakelists}" _content)
    string(
      REGEX REPLACE
      "\\$\\{CMAKE_SOURCE_DIR\\}(/[a-zA-Z_/]+)"
      "\${CMAKE_CURRENT_SOURCE_DIR}\\1"
      _content
      "${_content}"
    )
    file(WRITE "${_cmakelists}" "${_content}")
    message(VERBOSE "HipfortDependency: Patched hipfort CMakeLists.txt")
  endif()
endif()

set(_saved_install_prefix "${CMAKE_INSTALL_PREFIX}")
set(_hipfort_install_dir "${CMAKE_CURRENT_BINARY_DIR}/_hipfort")

block()
  set(CMAKE_INSTALL_PREFIX "${_hipfort_install_dir}" CACHE PATH "" FORCE)
  set(HIPFORT_INSTALL_DIR "${_hipfort_install_dir}" CACHE PATH "" FORCE)
  set(HIPFORT_COMPILER "${CMAKE_Fortran_COMPILER}" CACHE STRING "" FORCE)

  # hipfort's .f files are free-form with preprocessor directives
  # gfortran treats .f as fixed-form by default, so we need these flags
  # Also need unlimited line length as hipfort has very long lines
  set(_hipfort_flags "${CMAKE_Fortran_FLAGS}")
  if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    string(APPEND _hipfort_flags " -ffree-form -cpp -ffree-line-length-none")
  elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
    string(APPEND _hipfort_flags " -free -fpp")
  endif()
  set(HIPFORT_COMPILER_FLAGS "${_hipfort_flags}" CACHE STRING "" FORCE)

  if(CMAKE_Fortran_COMPILER_AR)
    set(HIPFORT_AR "${CMAKE_Fortran_COMPILER_AR}" CACHE STRING "" FORCE)
  endif()
  if(CMAKE_Fortran_COMPILER_RANLIB)
    set(HIPFORT_RANLIB "${CMAKE_Fortran_COMPILER_RANLIB}" CACHE STRING "" FORCE)
  endif()
  if(CMAKE_BUILD_TYPE)
    set(HIPFORT_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING "" FORCE)
  endif()

  add_subdirectory("${hipfort_SOURCE_DIR}" "${hipfort_BINARY_DIR}" EXCLUDE_FROM_ALL)
endblock()

set(CMAKE_INSTALL_PREFIX "${_saved_install_prefix}" CACHE PATH "" FORCE)

# hipfort creates hipfort-hip (real) and hipfort::hip (alias)
if(NOT TARGET hipfort::hip AND NOT TARGET hipfort-hip)
  message(
    FATAL_ERROR
    "HipfortDependency: Build failed - hipfort target not created.\n"
    "Check the build output above for errors."
  )
endif()

# Set SYSTEM on the real target (not the alias)
if(TARGET hipfort-hip)
  get_target_property(_type hipfort-hip TYPE)
  if(NOT _type STREQUAL "INTERFACE_LIBRARY")
    set_target_properties(hipfort-hip PROPERTIES SYSTEM TRUE)
  endif()
endif()

# Create hipfort::hipfort alias pointing to the real target
if(NOT TARGET hipfort::hipfort)
  if(TARGET hipfort-hip)
    add_library(hipfort::hipfort ALIAS hipfort-hip)
  elseif(TARGET hipfort::hip)
    # hipfort::hip exists but is already an alias - find the real target
    get_target_property(_aliased hipfort::hip ALIASED_TARGET)
    if(_aliased)
      add_library(hipfort::hipfort ALIAS ${_aliased})
    endif()
  endif()
endif()

set(HIPFORT_PROVIDER "fetched" CACHE INTERNAL "")
set(HIPFORT_VERSION "${HIPFORT_GIT_TAG}" CACHE INTERNAL "")

message(STATUS "HipfortDependency: Built hipfort ${HIPFORT_GIT_TAG} from source")
