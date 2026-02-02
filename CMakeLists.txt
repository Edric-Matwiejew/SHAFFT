cmake_minimum_required(VERSION 3.21)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "In-source builds are not allowed.")
endif()

project(shafft VERSION 0.1.0 DESCRIPTION "N-D HIP-accelerated MPI-parallel FFT." LANGUAGES CXX)

set(SHAFFT_VERSION_SUFFIX "-alpha" CACHE STRING "version suffix")

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

# Set CMP0144 to the new find_package behavior for _ROOT environment variables
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

option(SHAFFT_ENABLE_HIPFFT "Enable HIPFFT backend (GPU)" ON)
option(SHAFFT_ENABLE_FFTW "Enable FFTW backend (CPU)" OFF)

if(SHAFFT_ENABLE_FFTW)
  set(SHAFFT_ENABLE_HIPFFT OFF CACHE BOOL "" FORCE)
elseif(SHAFFT_ENABLE_HIPFFT)
  set(SHAFFT_ENABLE_FFTW OFF CACHE BOOL "" FORCE)
endif()

if(NOT SHAFFT_ENABLE_HIPFFT AND NOT SHAFFT_ENABLE_FFTW)
  message(STATUS "No backend specified, defaulting to HIPFFT")
  set(SHAFFT_ENABLE_HIPFFT ON CACHE BOOL "" FORCE)
endif()

# FFTW threading backend: "pthreads" (default) or "openmp"
set(SHAFFT_FFTW_THREADS "pthreads" CACHE STRING "FFTW threading backend: pthreads or openmp")
set_property(CACHE SHAFFT_FFTW_THREADS PROPERTY STRINGS pthreads openmp)

# GPU architecture override (optional - leave empty for auto-detect)
set(
  SHAFFT_GPU_ARCHITECTURES
  ""
  CACHE STRING
  "GPU architectures to compile for (e.g., gfx90a;gfx942 or sm_80;sm_90). Empty = auto-detect"
)

# Backend name + preprocessor defines visible to all subdirs
if(SHAFFT_ENABLE_HIPFFT)
  set(SHAFFT_BACKEND_NAME "hipFFT")
  set(SHAFFT_BACKEND_HIPFFT 1)
  set(SHAFFT_BACKEND_FFTW 0)
else()
  set(SHAFFT_BACKEND_NAME "FFTW")
  set(SHAFFT_BACKEND_HIPFFT 0)
  set(SHAFFT_BACKEND_FFTW 1)
endif()

# GPU-aware MPI: pass device pointers directly to MPI (requires compatible MPI)
option(SHAFFT_GPU_AWARE_MPI "Assume GPU-aware MPI for device pointer exchanges" ON)
if(SHAFFT_ENABLE_HIPFFT AND SHAFFT_GPU_AWARE_MPI)
  set(SHAFFT_GPU_AWARE_MPI_VAL 1)
  message(STATUS "GPU-aware MPI: enabled")
elseif(SHAFFT_ENABLE_HIPFFT)
  set(SHAFFT_GPU_AWARE_MPI_VAL 0)
  message(STATUS "GPU-aware MPI: disabled")
else()
  set(SHAFFT_GPU_AWARE_MPI_VAL 0) # Not applicable for FFTW
endif()

# ---------------- Generated config header ----------------
set(SHAFFT_CONFIG_BUILD_DIR ${PROJECT_BINARY_DIR}/include/shafft)
file(MAKE_DIRECTORY "${SHAFFT_CONFIG_BUILD_DIR}")
set(SHAFFT_CONFIG_HEADER "${SHAFFT_CONFIG_BUILD_DIR}/shafft_config.h")

# Guard against stale source-tree config headers that would shadow the generated one
if(EXISTS ${PROJECT_SOURCE_DIR}/include/shafft/shafft_config.h)
  message(
    FATAL_ERROR
    "Remove source-tree include/shafft/shafft_config.h; config is generated at build/include/shafft/shafft_config.h"
  )
endif()

configure_file(
  ${CMAKE_SOURCE_DIR}/cmake/shafft_config.h.in
  ${SHAFFT_CONFIG_BUILD_DIR}/shafft_config.h
  @ONLY
)

# ---------------- Common deps (all backends) ----------------
find_package(MPI REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type (Debug or Release)" FORCE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "Configuring for Debug build")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -O0 -g")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  message(STATUS "Configuring for Release build")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif()

# ---------------- Backend-specific toolchains & flags ----------------
if(SHAFFT_ENABLE_HIPFFT)
  if(SHAFFT_GPU_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES ${SHAFFT_GPU_ARCHITECTURES})
    message(STATUS "GPU architectures: ${CMAKE_HIP_ARCHITECTURES}")
  endif()

  # Enable HIP language
  enable_language(HIP)

  # Help CMake find ROCm cmake configs (needed for ROCm 6.x)
  if(DEFINED ENV{ROCM_PATH})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{ROCM_PATH}" "$ENV{ROCM_PATH}/lib/cmake")
  elseif(IS_DIRECTORY "/opt/rocm")
    list(APPEND CMAKE_PREFIX_PATH "/opt/rocm" "/opt/rocm/lib/cmake")
  endif()

  find_package(hip REQUIRED CONFIG)
  find_package(hipfft REQUIRED)

  # -------- gpuTT (GPU tensor transpose library) --------
  # Required for high-dimensional FFT performance (transpose-based algorithm)
  include(GputtDependency)
  message(STATUS "gpuTT: provider=${GPUTT_PROVIDER}, version=${GPUTT_VERSION}")

  if(SHAFFT_GPU_AWARE_MPI)
    option(SHAFFT_REQUIRE_GTL "Require GTL library for GPU-aware MPI on Cray systems" OFF)
    find_package(CrayGTL QUIET)
    if(CrayGTL_FOUND)
      set(SHAFFT_HAVE_GTL TRUE)
      message(STATUS "GTL library found: ${CrayGTL_LIBRARIES} (${CrayGTL_TYPE})")
    else()
      set(SHAFFT_HAVE_GTL FALSE)
      if(SHAFFT_REQUIRE_GTL)
        message(FATAL_ERROR "GTL library required but not found.")
      else()
        message(STATUS "GTL library not found")
      endif()
    endif()
  else()
    set(SHAFFT_HAVE_GTL FALSE)
  endif()

  # Initialize CMAKE_HIP_FLAGS
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

  # Add GCC toolchain path for HIP compiler
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    string(FIND "${CMAKE_HIP_FLAGS}" "--gcc-toolchain" GCC_TOOLCHAIN_FOUND)
    if(GCC_TOOLCHAIN_FOUND EQUAL -1)
      get_filename_component(CXX_COMPILER_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)
      get_filename_component(GCC_TOOLCHAIN_DIR ${CXX_COMPILER_DIR} DIRECTORY)
      set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --gcc-toolchain=${GCC_TOOLCHAIN_DIR}")
      message(STATUS "GCC toolchain: ${GCC_TOOLCHAIN_DIR}")
    endif()
  endif()

  message(STATUS "CMAKE_HIP_FLAGS: ${CMAKE_HIP_FLAGS}")
else() # SHAFFT_ENABLE_FFTW
  # -------- FFTW backend discovery (CPU) --------
  # Only require OpenMP when explicitly requested or when falling back from
  # pthreads to OpenMP threading because pthread libs are unavailable.
  set(_shafft_need_openmp FALSE)
  if(SHAFFT_FFTW_THREADS STREQUAL "openmp")
    set(_shafft_need_openmp TRUE)
    find_package(OpenMP REQUIRED)
  endif()

  # Fetch the FindFFTW.cmake module from GitHub
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
        master # upstream only publishes master/testing branches
      GIT_SHALLOW TRUE
      UPDATE_DISCONNECTED TRUE
    )
    FetchContent_MakeAvailable(FindFFTW)
  endif()

  list(APPEND CMAKE_MODULE_PATH "${findfftw_SOURCE_DIR}")

  # Find base libs (required): double + float
  find_package(FFTW REQUIRED COMPONENTS DOUBLE_LIB FLOAT_LIB)

  # Determine which threading targets to use based on user choice
  set(SHAFFT_FFTW_USE_OMP OFF)
  if(SHAFFT_FFTW_THREADS STREQUAL "openmp")
    if(TARGET FFTW::DoubleOpenMP AND TARGET FFTW::FloatOpenMP)
      set(SHAFFT_FFTW_USE_OMP ON)
    else()
      message(
        FATAL_ERROR
        "FFTW OpenMP threading requested but FFTW::DoubleOpenMP or FFTW::FloatOpenMP not found."
      )
    endif()
  else()
    # pthreads (default)
    if(NOT TARGET FFTW::DoubleThreads OR NOT TARGET FFTW::FloatThreads)
      # Fall back to OpenMP if pthreads not available
      if(TARGET FFTW::DoubleOpenMP AND TARGET FFTW::FloatOpenMP)
        set(SHAFFT_FFTW_USE_OMP ON)
        if(NOT _shafft_need_openmp)
          find_package(OpenMP REQUIRED)
        endif()
        message(STATUS "FFTW pthreads not found; using OpenMP threading")
      else()
        message(
          FATAL_ERROR
          "FFTW threading libraries not found (need either pthreads or OpenMP for both double and float)."
        )
      endif()
    endif()
  endif()

  message(STATUS "FFTW include dirs: ${FFTW_INCLUDE_DIRS}")
  message(STATUS "FFTW double lib : ${FFTW_DOUBLE_LIB}")
  message(STATUS "FFTW float  lib : ${FFTW_FLOAT_LIB}")
  if(SHAFFT_FFTW_USE_OMP)
    message(
      STATUS
      "FFTW threads    : OpenMP (double: ${FFTW_DOUBLE_OPENMP_LIB}, float: ${FFTW_FLOAT_OPENMP_LIB})"
    )
  else()
    message(
      STATUS
      "FFTW threads    : pthreads (double: ${FFTW_DOUBLE_THREADS_LIB}, float: ${FFTW_FLOAT_THREADS_LIB})"
    )
  endif()
endif()

# ---------------- Common INTERFACE library ----------------
add_library(shafft_options INTERFACE)
add_library(shafft::options ALIAS shafft_options)

target_compile_definitions(
  shafft_options
  INTERFACE
    SHAFFT_BACKEND_HIPFFT=${SHAFFT_BACKEND_HIPFFT}
    SHAFFT_BACKEND_FFTW=${SHAFFT_BACKEND_FFTW}
)

target_include_directories(
  shafft_options
  INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(SHAFFT_ENABLE_HIPFFT)
  target_link_libraries(shafft_options INTERFACE hip::host)
endif()

target_compile_options(
  shafft_options
  INTERFACE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-Wall -Wextra>
)

# ---------------- Subdirectories ----------------
add_subdirectory(src/cpp)
add_subdirectory(src/c)

# Optional Fortran interface
option(SHAFFT_BUILD_FORTRAN "Build Fortran 2003 interface" OFF)

if(SHAFFT_BUILD_FORTRAN)
  message(STATUS "Build Fortran - yes")
  # Set module directory here so it's available for install commands
  set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/include)
  add_subdirectory(src/f03)
else()
  message(STATUS "Build Fortran - no")
endif()

# Optional examples
option(SHAFFT_BUILD_EXAMPLES "Build example programs" OFF)

if(SHAFFT_BUILD_EXAMPLES)
  message(STATUS "Build examples - yes")
  add_subdirectory(examples)
else()
  message(STATUS "Build examples - no")
endif()

# Optional test suite (use CMake standard BUILD_TESTING)
include(CTest)
if(BUILD_TESTING)
  message(STATUS "Build tests   - yes")
  add_subdirectory(tests)
else()
  message(STATUS "Build tests   - no")
endif()

# ============================================================================
# Installation
# ============================================================================
include(CMakePackageConfigHelpers)

# Generate package config file
configure_package_config_file(
  ${CMAKE_SOURCE_DIR}/cmake/shafftConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/shafftConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/shafft
)

# Generate version file
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/shafftConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

# Install libraries
install(
  TARGETS shafftc++ shafftc shafft_options
  EXPORT shafftTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install Fortran library if built
if(SHAFFT_BUILD_FORTRAN)
  install(TARGETS shafftf03 EXPORT shafftTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(
    FILES ${CMAKE_Fortran_MODULE_DIRECTORY}/shafft.mod
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/shafft
  )
endif()

# Install public headers
install(
  DIRECTORY ${PROJECT_SOURCE_DIR}/include/shafft
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN "*.hpp"
  PATTERN "*.h"
)

# Install generated config header
install(FILES ${SHAFFT_CONFIG_HEADER} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/shafft)

# Install package config files
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/shafftConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/shafftConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/shafft
)

# Export targets for find_package()
install(EXPORT shafftTargets NAMESPACE shafft:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/shafft)
