cmake_minimum_required(VERSION 3.25)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "In-source builds are not allowed.")
endif()

project(shafft VERSION 1.1.0 DESCRIPTION "N-D HIP-accelerated MPI-parallel FFT." LANGUAGES CXX)

set(SHAFFT_VERSION_SUFFIX "-alpha" CACHE STRING "version suffix")

set(SHAFFT_DEPS_DIR "${CMAKE_SOURCE_DIR}/.deps" CACHE PATH "Shared directory for downloaded dependencies")
set(FETCHCONTENT_BASE_DIR "${SHAFFT_DEPS_DIR}" CACHE PATH "FetchContent base directory" FORCE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
if(APPLE)
  set(CMAKE_INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
else()
  set(CMAKE_INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()

if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

include(Tooling)

option(SHAFFT_ENABLE_HIPFFT "Enable HIPFFT backend (GPU)" ON)
option(SHAFFT_ENABLE_FFTW "Enable FFTW backend (CPU)" OFF)

if(SHAFFT_ENABLE_FFTW)
  set(SHAFFT_ENABLE_HIPFFT OFF CACHE BOOL "" FORCE)
elseif(SHAFFT_ENABLE_HIPFFT)
  set(SHAFFT_ENABLE_FFTW OFF CACHE BOOL "" FORCE)
endif()

if(NOT SHAFFT_ENABLE_HIPFFT AND NOT SHAFFT_ENABLE_FFTW)
  message(STATUS "SHAFFT Backend: hipFFT (default)")
  set(SHAFFT_ENABLE_HIPFFT ON CACHE BOOL "" FORCE)
endif()

set(SHAFFT_FFTW_THREADS "pthreads" CACHE STRING "FFTW threading backend: pthreads or openmp")
set_property(CACHE SHAFFT_FFTW_THREADS PROPERTY STRINGS pthreads openmp)

set(
  SHAFFT_GPU_ARCHITECTURES
  ""
  CACHE STRING
  "GPU architectures (e.g., gfx90a;gfx942 or sm_80;sm_90); empty = auto-detect"
)

if(SHAFFT_ENABLE_HIPFFT)
  set(SHAFFT_BACKEND_NAME "hipFFT")
  set(SHAFFT_BACKEND_HIPFFT 1)
  set(SHAFFT_BACKEND_FFTW 0)
else()
  set(SHAFFT_BACKEND_NAME "FFTW")
  set(SHAFFT_BACKEND_HIPFFT 0)
  set(SHAFFT_BACKEND_FFTW 1)
endif()

option(SHAFFT_GPU_AWARE_MPI "Assume GPU-aware MPI for device pointer exchanges" ON)
if(SHAFFT_ENABLE_HIPFFT AND SHAFFT_GPU_AWARE_MPI)
  set(SHAFFT_GPU_AWARE_MPI_VAL 1)
  message(STATUS "SHAFFT GPU-aware MPI: on")
elseif(SHAFFT_ENABLE_HIPFFT)
  set(SHAFFT_GPU_AWARE_MPI_VAL 0)
  message(STATUS "SHAFFT GPU-aware MPI: off")
else()
  set(SHAFFT_GPU_AWARE_MPI_VAL 0)
endif()

set(SHAFFT_CONFIG_BUILD_DIR ${PROJECT_BINARY_DIR}/include/shafft)
file(MAKE_DIRECTORY "${SHAFFT_CONFIG_BUILD_DIR}")
set(SHAFFT_CONFIG_HEADER "${SHAFFT_CONFIG_BUILD_DIR}/shafft_config.h")

if(EXISTS ${PROJECT_SOURCE_DIR}/include/shafft/shafft_config.h)
  message(FATAL_ERROR "Remove source-tree include/shafft/shafft_config.h (generated in build)")
endif()

configure_file(
  ${CMAKE_SOURCE_DIR}/cmake/shafft_config.h.in
  ${SHAFFT_CONFIG_BUILD_DIR}/shafft_config.h
  @ONLY
)

find_package(MPI REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type (Debug or Release)" FORCE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "Build type: Debug")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -O0 -g")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  message(STATUS "Build type: Release")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif()

if(SHAFFT_ENABLE_HIPFFT)
  if(SHAFFT_GPU_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES ${SHAFFT_GPU_ARCHITECTURES})
    message(STATUS "GPU architectures: ${CMAKE_HIP_ARCHITECTURES}")
  endif()

  enable_language(HIP)

  if(DEFINED ENV{ROCM_PATH})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{ROCM_PATH}" "$ENV{ROCM_PATH}/lib/cmake")
  elseif(IS_DIRECTORY "/opt/rocm")
    list(APPEND CMAKE_PREFIX_PATH "/opt/rocm" "/opt/rocm/lib/cmake")
  endif()

  find_package(hip REQUIRED CONFIG)
  find_package(hipfft REQUIRED)

  include(GputtDependency)
  message(STATUS "gpuTT: provider=${GPUTT_PROVIDER}, version=${GPUTT_VERSION}")

  if(SHAFFT_GPU_AWARE_MPI)
    option(SHAFFT_REQUIRE_GTL "Require GTL library for GPU-aware MPI on Cray systems" OFF)
    find_package(CrayGTL QUIET)
    if(CrayGTL_FOUND)
      set(SHAFFT_HAVE_GTL TRUE)
      message(STATUS "GTL: ${CrayGTL_LIBRARIES} (${CrayGTL_TYPE})")
    else()
      set(SHAFFT_HAVE_GTL FALSE)
      if(SHAFFT_REQUIRE_GTL)
        message(FATAL_ERROR "GTL required but not found")
      else()
        message(STATUS "GTL not found")
      endif()
    endif()
  else()
    set(SHAFFT_HAVE_GTL FALSE)
  endif()

  if(NOT DEFINED CMAKE_HIP_FLAGS)
    set(CMAKE_HIP_FLAGS "")
  endif()

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -g -O0")
  elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O3")
  endif()

  string(FIND "${CMAKE_HIP_FLAGS}" "-fPIC" PIC_FOUND)
  if(PIC_FOUND EQUAL -1)
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -fPIC")
  endif()

  set(SHAFFT_HIP_GCC_TOOLCHAIN "" CACHE PATH
    "GCC toolchain root for HIP compiler (auto-detected if empty)")
  
  string(FIND "${CMAKE_HIP_FLAGS}" "--gcc-toolchain" GCC_TOOLCHAIN_FOUND)
  if(GCC_TOOLCHAIN_FOUND EQUAL -1)
    if(SHAFFT_HIP_GCC_TOOLCHAIN)
      set(_gcc_toolchain "${SHAFFT_HIP_GCC_TOOLCHAIN}")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      get_filename_component(_cxx_dir ${CMAKE_CXX_COMPILER} DIRECTORY)
      get_filename_component(_gcc_toolchain ${_cxx_dir} DIRECTORY)
    endif()
    
    if(_gcc_toolchain AND IS_DIRECTORY "${_gcc_toolchain}")
      set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --gcc-toolchain=${_gcc_toolchain}")
      message(STATUS "GCC toolchain: ${_gcc_toolchain}")
    endif()
    
    execute_process(
      COMMAND ${CMAKE_CXX_COMPILER} -dumpversion
      OUTPUT_VARIABLE _gcc_version
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(REGEX MATCH "^[0-9]+" _gcc_major "${_gcc_version}")
    
    set(_cxx_inc "")
    foreach(_base IN ITEMS "${_gcc_toolchain}/include/c++" "/usr/include/c++")
      if(IS_DIRECTORY "${_base}/${_gcc_major}")
        set(_cxx_inc "${_base}/${_gcc_major}")
        break()
      elseif(IS_DIRECTORY "${_base}/${_gcc_version}")
        set(_cxx_inc "${_base}/${_gcc_version}")
        break()
      endif()
    endforeach()
    
    if(_cxx_inc)
      set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -isystem ${_cxx_inc}")
      message(STATUS "GCC C++ includes: ${_cxx_inc}")
      
      file(GLOB _target_dirs "${_cxx_inc}/*-linux*")
      foreach(_tdir IN LISTS _target_dirs)
        if(IS_DIRECTORY "${_tdir}")
          set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -isystem ${_tdir}")
          message(STATUS "GCC C++ target includes: ${_tdir}")
        endif()
      endforeach()
    endif()
  endif()

  message(STATUS "CMAKE_HIP_FLAGS: ${CMAKE_HIP_FLAGS}")
else()
  set(_shafft_need_openmp FALSE)
  if(SHAFFT_FFTW_THREADS STREQUAL "openmp")
    set(_shafft_need_openmp TRUE)
    find_package(OpenMP REQUIRED)
  endif()

  option(
    SHAFFT_FINDFFTW_SOURCE_DIR
    "Path to a local checkout of findfftw (contains FindFFTW.cmake)"
    ""
  )

  if(SHAFFT_FINDFFTW_SOURCE_DIR)
    set(findfftw_SOURCE_DIR "${SHAFFT_FINDFFTW_SOURCE_DIR}")
  else()
    include(FetchContent)
    FetchContent_Declare(
      FindFFTW
      GIT_REPOSITORY https://github.com/egpbos/findfftw.git
      GIT_TAG
        master
      GIT_SHALLOW TRUE
      UPDATE_DISCONNECTED TRUE
    )
    FetchContent_MakeAvailable(FindFFTW)
  endif()

  list(APPEND CMAKE_MODULE_PATH "${findfftw_SOURCE_DIR}")

  find_package(FFTW REQUIRED COMPONENTS DOUBLE_LIB FLOAT_LIB)

  set(SHAFFT_FFTW_USE_OMP OFF)
  if(SHAFFT_FFTW_THREADS STREQUAL "openmp")
    if(TARGET FFTW::DoubleOpenMP AND TARGET FFTW::FloatOpenMP)
      set(SHAFFT_FFTW_USE_OMP ON)
    else()
      message(FATAL_ERROR "FFTW OpenMP requested but OpenMP libs not found")
    endif()
  else()
    if(NOT TARGET FFTW::DoubleThreads OR NOT TARGET FFTW::FloatThreads)
      if(TARGET FFTW::DoubleOpenMP AND TARGET FFTW::FloatOpenMP)
        set(SHAFFT_FFTW_USE_OMP ON)
        if(NOT _shafft_need_openmp)
          find_package(OpenMP REQUIRED)
        endif()
        message(STATUS "FFTW pthreads not found; using OpenMP threading")
      else()
        message(FATAL_ERROR "FFTW threads libs missing (need pthreads or OpenMP)")
      endif()
    endif()
  endif()

  message(STATUS "FFTW include dirs: ${FFTW_INCLUDE_DIRS}")
  message(STATUS "FFTW double lib : ${FFTW_DOUBLE_LIB}")
  message(STATUS "FFTW float  lib : ${FFTW_FLOAT_LIB}")
  if(SHAFFT_FFTW_USE_OMP)
    message(STATUS "FFTW threads    : OpenMP (double: ${FFTW_DOUBLE_OPENMP_LIB}, float: ${FFTW_FLOAT_OPENMP_LIB})")
  else()
    message(STATUS "FFTW threads    : pthreads (double: ${FFTW_DOUBLE_THREADS_LIB}, float: ${FFTW_FLOAT_THREADS_LIB})")
  endif()
endif()

add_library(shafft_options INTERFACE)
add_library(shafft::options ALIAS shafft_options)

target_compile_definitions(
  shafft_options
  INTERFACE
    SHAFFT_BACKEND_HIPFFT=${SHAFFT_BACKEND_HIPFFT}
    SHAFFT_BACKEND_FFTW=${SHAFFT_BACKEND_FFTW}
    SHAFFT_GPU_AWARE_MPI=${SHAFFT_GPU_AWARE_MPI_VAL}
)

target_include_directories(
  shafft_options
  INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(SHAFFT_ENABLE_HIPFFT)
  target_link_libraries(shafft_options INTERFACE hip::host)
endif()

target_compile_options(
  shafft_options
  INTERFACE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-Wall -Wextra>
)

add_subdirectory(src/cpp)
add_subdirectory(src/c)

option(SHAFFT_BUILD_FORTRAN "Build Fortran 2003 interface" OFF)

if(SHAFFT_BUILD_FORTRAN)
  message(STATUS "SHAFFT Fortran: on")
  set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/include)
  add_subdirectory(src/f03)
else()
  message(STATUS "SHAFFT Fortran: off")
endif()

option(SHAFFT_BUILD_EXAMPLES "Build example programs" OFF)

if(SHAFFT_BUILD_EXAMPLES)
  message(STATUS "SHAFFT Examples: on")
  add_subdirectory(examples)
else()
  message(STATUS "SHAFFT Examples: off")
endif()

option(SHAFFT_BUILD_TESTING "Build SHAFFT test suite" ON)

include(CTest)
if(SHAFFT_BUILD_TESTING AND BUILD_TESTING)
  message(STATUS "SHAFFT Tests: on")
  add_subdirectory(tests)
else()
  if(NOT SHAFFT_BUILD_TESTING)
    message(STATUS "SHAFFT Tests: off (SHAFFT_BUILD_TESTING=OFF)")
  else()
    message(STATUS "SHAFFT Tests: off (BUILD_TESTING=OFF)")
  endif()
endif()

include(CMakePackageConfigHelpers)

set(_shafft_cmake_builddir "${CMAKE_CURRENT_BINARY_DIR}/cmake-pkg-install")
file(MAKE_DIRECTORY "${_shafft_cmake_builddir}")

configure_package_config_file(
  ${CMAKE_SOURCE_DIR}/cmake/shafftConfig.cmake.in
  ${_shafft_cmake_builddir}/shafftConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/shafft
  PATH_VARS CMAKE_INSTALL_INCLUDEDIR
)

write_basic_package_version_file(
  ${_shafft_cmake_builddir}/shafftConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(
  TARGETS shafftc++ shafftc shafft_options
  EXPORT shafftTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(SHAFFT_BUILD_FORTRAN)
  install(TARGETS shafftf03 EXPORT shafftTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(
    FILES ${CMAKE_Fortran_MODULE_DIRECTORY}/shafft.mod
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/shafft
  )
endif()

install(
  DIRECTORY ${PROJECT_SOURCE_DIR}/include/shafft
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN "*.hpp"
  PATTERN "*.h"
)

install(FILES ${SHAFFT_CONFIG_HEADER} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/shafft)

install(
  FILES
    ${_shafft_cmake_builddir}/shafftConfig.cmake
    ${_shafft_cmake_builddir}/shafftConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/shafft
)

install(EXPORT shafftTargets NAMESPACE shafft:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/shafft)
