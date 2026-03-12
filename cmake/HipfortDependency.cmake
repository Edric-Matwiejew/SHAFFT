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

option(HIPFORT_AUTO_FETCH
  "Fetch and build if find_package fails" ON)

option(HIPFORT_VERIFY_MOD_COMPAT
  "Verify .mod compatibility with Fortran compiler" ON)

option(HIPFORT_REQUIRE_ROCM
  "Require ROCm installation" ON)

set(HIPFORT_GIT_TAG "" CACHE STRING
  "Git tag/branch; auto-detected from ROCm if empty")

set(HIPFORT_GIT_REPO "https://github.com/ROCm/hipfort.git" CACHE STRING
  "Git repository URL")

set(HIPFORT_SOURCE_URL "" CACHE STRING
  "Source URL or path; overrides git")

set(HIPFORT_SOURCE_URL_HASH "" CACHE STRING
  "URL hash (e.g. SHA256=...)")

mark_as_advanced(
  HIPFORT_AUTO_FETCH
  HIPFORT_VERIFY_MOD_COMPAT
  HIPFORT_REQUIRE_ROCM
  HIPFORT_GIT_TAG
  HIPFORT_GIT_REPO
  HIPFORT_SOURCE_URL
  HIPFORT_SOURCE_URL_HASH
)

#=============================================================================
# Helper function and macro
#=============================================================================

# Check .mod file compatibility with current Fortran compiler
function(_hipfort_check_mod_compatibility target result_var)
  if(TARGET hipfort::hipfort-amdgcn)
    get_target_property(_inc_dirs hipfort::hipfort-amdgcn INTERFACE_INCLUDE_DIRECTORIES)
  elseif(TARGET hipfort::hipfort-nvptx)
    get_target_property(_inc_dirs hipfort::hipfort-nvptx INTERFACE_INCLUDE_DIRECTORIES)
  else()
    get_target_property(_inc_dirs ${target} INTERFACE_INCLUDE_DIRECTORIES)
  endif()

  set(_mod_flags "")
  foreach(_dir IN LISTS _inc_dirs)
    if(_dir AND NOT _dir MATCHES "^\\$<" AND IS_DIRECTORY "${_dir}")
      string(APPEND _mod_flags " -I${_dir}")
    endif()
  endforeach()

  try_compile(_compat_ok
    SOURCE_FROM_CONTENT _hipfort_check.f90
      "program p; use hipfort; end program"
    CMAKE_FLAGS
      "-DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}"
      "-DCMAKE_Fortran_FLAGS=${CMAKE_Fortran_FLAGS}${_mod_flags}"
    OUTPUT_VARIABLE _compat_log
  )

  set(${result_var} ${_compat_ok} PARENT_SCOPE)
  set(${result_var}_LOG ${_compat_log} PARENT_SCOPE)
endfunction()

# Ensure hipfort::hipfort alias exists
macro(_hipfort_ensure_alias)
  if(NOT TARGET hipfort::hipfort AND TARGET hipfort::hip)
    add_library(hipfort::hipfort ALIAS hipfort::hip)
  endif()
endmacro()

#=============================================================================
# Validate prerequisites
#=============================================================================

if(NOT CMAKE_Fortran_COMPILER)
  message(FATAL_ERROR
    "HipfortDependency: Fortran compiler not found. "
    "Call project(... LANGUAGES Fortran) before including this module.")
endif()

if(NOT CMAKE_C_COMPILER_LOADED AND NOT CMAKE_CXX_COMPILER_LOADED)
  enable_language(C)
endif()

#=============================================================================
# ROCm detection
#=============================================================================

set(HIPFORT_ROCM_ROOT "" CACHE PATH
  "ROCm installation root; auto-detected if empty")
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
  message(FATAL_ERROR
    "HipfortDependency: ROCm not found. Options:\n"
    "  - Set ROCM_PATH to your ROCm installation\n"
    "  - Add ROCm to CMAKE_PREFIX_PATH\n"
    "  - Set HIPFORT_REQUIRE_ROCM=OFF if ROCm is not needed")
endif()

if(HIPFORT_ROCM_ROOT)
  list(PREPEND CMAKE_PREFIX_PATH
    "${HIPFORT_ROCM_ROOT}"
    "${HIPFORT_ROCM_ROOT}/lib/cmake/hipfort"
  )
  message(VERBOSE "HipfortDependency: ROCm root = ${HIPFORT_ROCM_ROOT}")
endif()

#=============================================================================
# ROCm version detection (for auto-tagging)
#=============================================================================

set(HIPFORT_ROCM_VERSION "" CACHE STRING "Detected ROCm version" FORCE)
mark_as_advanced(HIPFORT_ROCM_VERSION)

if(HIPFORT_ROCM_ROOT AND NOT HIPFORT_ROCM_VERSION)
  set(_version_files
    "${HIPFORT_ROCM_ROOT}/.info/version"
    "${HIPFORT_ROCM_ROOT}/lib/cmake/hip/hip-config-version.cmake"
  )
  foreach(_vfile IN LISTS _version_files)
    if(EXISTS "${_vfile}")
      file(READ "${_vfile}" _ver_content)
      if(_ver_content MATCHES "([0-9]+\\.[0-9]+\\.[0-9]+)")
        set(HIPFORT_ROCM_VERSION "${CMAKE_MATCH_1}" CACHE STRING "" FORCE)
        break()
      endif()
    endif()
  endforeach()

  if(HIPFORT_ROCM_VERSION)
    message(VERBOSE "HipfortDependency: ROCm version = ${HIPFORT_ROCM_VERSION}")
  endif()
endif()

if(NOT HIPFORT_GIT_TAG)
  if(HIPFORT_ROCM_VERSION)
    set(HIPFORT_GIT_TAG "rocm-${HIPFORT_ROCM_VERSION}"
      CACHE STRING "Git tag/branch; auto-detected from ROCm if empty" FORCE)
  else()
    set(HIPFORT_GIT_TAG "rocm-6.2.0"
      CACHE STRING "Git tag/branch; auto-detected from ROCm if empty" FORCE)
    if(HIPFORT_REQUIRE_ROCM)
      message(WARNING
        "HipfortDependency: Could not detect ROCm version, using ${HIPFORT_GIT_TAG}")
    endif()
  endif()
endif()

#=============================================================================
# Try system hipfort
#=============================================================================

# Check user-specified path first (backward compatibility with HIPFORT_PATH)
if(HIPFORT_PATH)
  set(_hf_path "${HIPFORT_PATH}")
elseif(DEFINED ENV{HIPFORT_PATH})
  set(_hf_path "$ENV{HIPFORT_PATH}")
endif()
if(_hf_path)
  list(PREPEND CMAKE_PREFIX_PATH "${_hf_path}" "${_hf_path}/lib/cmake")
endif()

# Request 'hip' component - required for hipfort::hip target
find_package(hipfort CONFIG QUIET COMPONENTS hip)

if(hipfort_FOUND)
  message(VERBOSE "HipfortDependency: Found system hipfort at ${hipfort_DIR}")

  if(TARGET hipfort::hip)
    set(_hipfort_target hipfort::hip)
  elseif(TARGET hipfort::hipfort)
    set(_hipfort_target hipfort::hipfort)
  else()
    message(WARNING
      "HipfortDependency: Package found but targets missing; will fetch instead")
    set(hipfort_FOUND FALSE)
  endif()
endif()

if(hipfort_FOUND AND HIPFORT_VERIFY_MOD_COMPAT)
  _hipfort_check_mod_compatibility(${_hipfort_target} _mod_compat_ok)

  if(_mod_compat_ok)
    message(VERBOSE "HipfortDependency: Module compatibility check passed")
  else()
    message(STATUS
      "HipfortDependency: System hipfort .mod files incompatible with "
      "${CMAKE_Fortran_COMPILER_ID} ${CMAKE_Fortran_COMPILER_VERSION} "
      "(likely built with different compiler); will fetch and build instead")
    message(VERBOSE "Module compatibility check output:\n${_mod_compat_ok_LOG}")
    set(hipfort_FOUND FALSE)
  endif()
endif()

if(hipfort_FOUND)
  _hipfort_ensure_alias()

  set(HIPFORT_PROVIDER "system" CACHE INTERNAL "")
  set(HIPFORT_VERSION "${hipfort_VERSION}" CACHE INTERNAL "")

  message(STATUS "HipfortDependency: Using system hipfort ${hipfort_VERSION}")
  return()
endif()

#=============================================================================
# Check for previously-built hipfort in .deps (persists across clean builds)
#=============================================================================

set(_hipfort_install_dir "${FETCHCONTENT_BASE_DIR}/hipfort-install")
set(_hipfort_cached_config "${_hipfort_install_dir}/lib/cmake/hipfort/hipfort-config.cmake")

if(EXISTS "${_hipfort_cached_config}")
  message(STATUS "HipfortDependency: Found previously-built hipfort in ${_hipfort_install_dir}")

  set(hipfort_DIR "${_hipfort_install_dir}/lib/cmake/hipfort")
  find_package(hipfort CONFIG QUIET COMPONENTS hip)

  if(hipfort_FOUND)
    set(_use_cached TRUE)

    if(HIPFORT_VERIFY_MOD_COMPAT)
      _hipfort_check_mod_compatibility(hipfort::hip _cached_mod_compat_ok)

      if(NOT _cached_mod_compat_ok)
        message(STATUS "HipfortDependency: Cached hipfort .mod files incompatible with current compiler; will rebuild")
        message(DEBUG "HipfortDependency: try_compile output: ${_cached_mod_compat_ok_LOG}")
        set(_use_cached FALSE)
      endif()
    endif()

    if(_use_cached)
      message(STATUS "HipfortDependency: Using cached hipfort from ${_hipfort_install_dir}")
      _hipfort_ensure_alias()
      set(HIPFORT_PROVIDER "cached" CACHE INTERNAL "")
      set(HIPFORT_VERSION "${hipfort_VERSION}" CACHE INTERNAL "")
      return()
    endif()
  endif()
endif()

#=============================================================================
# Fetch and build hipfort
#=============================================================================

if(NOT HIPFORT_AUTO_FETCH)
  message(FATAL_ERROR
    "HipfortDependency: hipfort not found and HIPFORT_AUTO_FETCH is OFF.\n"
    "Options:\n"
    "  - Set hipfort_DIR to existing installation\n"
    "  - Add hipfort location to CMAKE_PREFIX_PATH\n"
    "  - Set HIPFORT_AUTO_FETCH=ON to download automatically")
endif()

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

  set(_fc_args URL "${_source_url}")
  if(HIPFORT_SOURCE_URL_HASH)
    list(APPEND _fc_args URL_HASH "${HIPFORT_SOURCE_URL_HASH}")
  endif()
else()
  set(_fc_args
    GIT_REPOSITORY "${HIPFORT_GIT_REPO}"
    GIT_TAG "${HIPFORT_GIT_TAG}"
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
  )
endif()

include(ExternalProject)

# Build hipfort compiler flags
set(_hipfort_flags "${CMAKE_Fortran_FLAGS}")
if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  string(APPEND _hipfort_flags " -ffree-form -cpp -ffree-line-length-none")
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
  string(APPEND _hipfort_flags " -free -fpp")
endif()

# Build optional arguments list - only add if defined
set(_optional_args "")

# GPU_TARGETS
if(DEFINED GPU_TARGETS)
  list(APPEND _optional_args "-DGPU_TARGETS=${GPU_TARGETS}")
elseif(DEFINED OFFLOAD_ARCH)
  list(APPEND _optional_args "-DGPU_TARGETS=${OFFLOAD_ARCH}")
endif()

# AR/RANLIB
if(CMAKE_Fortran_COMPILER_AR)
  list(APPEND _optional_args "-DHIPFORT_AR=${CMAKE_Fortran_COMPILER_AR}")
endif()
if(CMAKE_Fortran_COMPILER_RANLIB)
  list(APPEND _optional_args "-DHIPFORT_RANLIB=${CMAKE_Fortran_COMPILER_RANLIB}")
endif()

# ROCm path - hipfort needs this to find HIP headers
if(HIPFORT_ROCM_ROOT)
  list(APPEND _optional_args "-DROCM_PATH=${HIPFORT_ROCM_ROOT}")
endif()

# Determine platform suffix for library and include paths
if(HIP_PLATFORM STREQUAL "nvidia")
  set(_hipfort_platform "nvptx")
else()
  set(_hipfort_platform "amdgcn")
endif()

ExternalProject_Add(hipfort_external
  ${_fc_args}
  PREFIX "${FETCHCONTENT_BASE_DIR}/hipfort-ep"
  INSTALL_DIR "${_hipfort_install_dir}"
  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    -DHIPFORT_INSTALL_DIR=<INSTALL_DIR>
    -DHIPFORT_COMPILER=${CMAKE_Fortran_COMPILER}
    "-DHIPFORT_COMPILER_FLAGS=${_hipfort_flags}"
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DBUILD_TESTING=OFF
    ${_optional_args}
  BUILD_BYPRODUCTS
    "${_hipfort_install_dir}/lib/libhipfort-amdgcn.a"
    "${_hipfort_install_dir}/lib/libhipfort-nvptx.a"
)

# Create an imported target that depends on the external project
add_library(hipfort::hipfort INTERFACE IMPORTED GLOBAL)
add_dependencies(hipfort::hipfort hipfort_external)

# Set include directories - platform-specific path for .mod files
set_target_properties(hipfort::hipfort PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${_hipfort_install_dir}/include/hipfort/${_hipfort_platform}"
)

# Link to the hipfort library - platform-specific
set_property(TARGET hipfort::hipfort APPEND PROPERTY
  INTERFACE_LINK_LIBRARIES "${_hipfort_install_dir}/lib/libhipfort-${_hipfort_platform}.a")

set(HIPFORT_PROVIDER "fetched" CACHE INTERNAL "")
set(HIPFORT_VERSION "${HIPFORT_GIT_TAG}" CACHE INTERNAL "")

message(STATUS "HipfortDependency: Will build and install hipfort ${HIPFORT_GIT_TAG} to ${_hipfort_install_dir}")
