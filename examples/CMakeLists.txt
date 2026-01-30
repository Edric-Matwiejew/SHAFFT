cmake_minimum_required(VERSION 3.21)
project(shafft_examples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# Detect build mode: standalone vs subdirectory of main SHAFFT build
# ============================================================================
if(TARGET shafft::shafftc++)
  # Built as subdirectory - targets already available
  message(STATUS "Building examples as part of SHAFFT build")
  set(SHAFFT_EXAMPLES_STANDALONE FALSE)
else()
  # Standalone build - use find_package
  message(STATUS "Building examples standalone - using find_package(shafft)")
  set(SHAFFT_EXAMPLES_STANDALONE TRUE)

  if(NOT DEFINED SHAFFT_PATH)
    message(FATAL_ERROR "Please configure with -DSHAFFT_PATH=/path/to/shafft/install/prefix")
  endif()

  find_package(shafft REQUIRED
    HINTS "${SHAFFT_PATH}/lib/cmake/shafft"
          "${SHAFFT_PATH}/${CMAKE_INSTALL_LIBDIR}/cmake/shafft"
  )

  # Check which backend was used
  if(SHAFFT_BACKEND_HIPFFT)
    set(SHAFFT_USE_HIPFFT TRUE)
    message(STATUS "SHAFFT was built with HIPFFT backend")
  else()
    set(SHAFFT_USE_HIPFFT FALSE)
    message(STATUS "SHAFFT was built with FFTW backend")
  endif()
endif()

# For standalone builds, we still need FFTW for local FFT operations in examples
if(SHAFFT_EXAMPLES_STANDALONE)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(FFTW3  REQUIRED IMPORTED_TARGET fftw3)
  pkg_check_modules(FFTW3F REQUIRED IMPORTED_TARGET fftw3f)

  # Threads: try pkg-config first, then fall back to find_library
  set(FFTW_THREADS_TARGET "")
  pkg_check_modules(FFTW3_THREADS QUIET IMPORTED_TARGET fftw3_threads)
  if(FFTW3_THREADS_FOUND)
    set(FFTW_THREADS_TARGET PkgConfig::FFTW3_THREADS)
  else()
    pkg_check_modules(FFTW3_OMP QUIET IMPORTED_TARGET fftw3_omp)
    if(FFTW3_OMP_FOUND)
      set(FFTW_THREADS_TARGET PkgConfig::FFTW3_OMP)
    else()
      find_library(FFTW_THREADS_LIB
        NAMES fftw3_threads fftw3_omp
        HINTS /opt/homebrew/lib ENV LIBRARY_PATH ${CMAKE_PREFIX_PATH}
      )
      if(NOT FFTW_THREADS_LIB)
        message(FATAL_ERROR "Could not find FFTW threads library (fftw3_threads or fftw3_omp).")
      endif()
      set(FFTW_THREADS_TARGET "${FFTW_THREADS_LIB}")
    endif()
  endif()
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# ============================================================================
# HIP-specific example - only built when HIPFFT backend is used
# ============================================================================
if(SHAFFT_USE_HIPFFT OR SHAFFT_ENABLE_HIPFFT)
  add_executable(example_hip example.cpp)

  target_link_libraries(example_hip
    PRIVATE
      shafft::shafftc++
      MPI::MPI_CXX
      OpenMP::OpenMP_CXX
  )

  # Standalone builds need FFTW for local operations
  if(SHAFFT_EXAMPLES_STANDALONE)
    target_link_libraries(example_hip PRIVATE
      PkgConfig::FFTW3 PkgConfig::FFTW3F ${FFTW_THREADS_TARGET})
  endif()
endif()

# ============================================================================
# Portable example - demonstrates backend-agnostic API
# ============================================================================
add_executable(example_portable example_portable.cpp)

target_link_libraries(example_portable
  PRIVATE
    shafft::shafftc++
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
)

if(SHAFFT_EXAMPLES_STANDALONE)
  target_link_libraries(example_portable PRIVATE
    PkgConfig::FFTW3 PkgConfig::FFTW3F ${FFTW_THREADS_TARGET})
endif()

# ============================================================================
# C interface example - demonstrates C API with portable buffer functions
# ============================================================================
enable_language(C)

find_package(MPI COMPONENTS C REQUIRED)

add_executable(example_c example_c.c)

target_link_libraries(example_c
  PRIVATE
    shafft::shafftc
    MPI::MPI_C
    m  # math library for complex.h
)

if(SHAFFT_EXAMPLES_STANDALONE)
  target_link_libraries(example_c PRIVATE
    PkgConfig::FFTW3 PkgConfig::FFTW3F ${FFTW_THREADS_TARGET})
endif()

# ============================================================================
# Fortran interface example - portable version
# ============================================================================
if(SHAFFT_EXAMPLES_STANDALONE AND TARGET shafft::shafftf03)
  # Standalone build with installed Fortran library
  enable_language(Fortran)
  find_package(MPI COMPONENTS Fortran REQUIRED)

  add_executable(example_f03_portable example_portable.f03)

  target_link_libraries(example_f03_portable
    PRIVATE
      shafft::shafftf03
      MPI::MPI_Fortran
      PkgConfig::FFTW3
      PkgConfig::FFTW3F
      ${FFTW_THREADS_TARGET}
  )
elseif(NOT SHAFFT_EXAMPLES_STANDALONE AND SHAFFT_BUILD_FORTRAN)
  # Built as subdirectory with Fortran enabled - use real target name
  enable_language(Fortran)
  find_package(MPI COMPONENTS Fortran REQUIRED)

  add_executable(example_f03_portable example_portable.f03)

  # Need to include the directory where shafft.mod is generated
  target_include_directories(example_f03_portable PRIVATE
    ${CMAKE_BINARY_DIR}/include  # Location of shafft.mod
  )

  target_link_libraries(example_f03_portable
    PRIVATE
      shafftf03  # Real target name (not alias)
      MPI::MPI_Fortran
  )
endif()

# ============================================================================
# HIP-specific Fortran example - requires hipfort
# ============================================================================
# NOTE: This requires hipfort module files compiled with the same Fortran compiler.
#       The ROCm-provided hipfort.mod is typically flang-compiled and incompatible with gfortran.
#       To enable: rebuild hipfort with gfortran, or use hipfc/flang throughout.
option(SHAFFT_BUILD_HIPFORT_EXAMPLE "Build Fortran example using hipfort (requires compatible hipfort)" OFF)

if(SHAFFT_BUILD_HIPFORT_EXAMPLE)
  if(NOT (SHAFFT_USE_HIPFFT OR SHAFFT_ENABLE_HIPFFT))
    message(WARNING "SHAFFT_BUILD_HIPFORT_EXAMPLE=ON but HIPFFT backend not enabled - skipping example_f03_hip")
  else()
    enable_language(Fortran)
    find_package(MPI COMPONENTS Fortran REQUIRED)

    # Try to find hipfort - ROCm provides hipfort::hipfort-amdgcn target
    find_package(hipfort QUIET CONFIG)

    # Check for the actual target name (varies by platform)
    set(HIPFORT_TARGET "")
    if(TARGET hipfort::hipfort-amdgcn)
      set(HIPFORT_TARGET hipfort::hipfort-amdgcn)
    elseif(TARGET hipfort::hipfort-nvptx)
      set(HIPFORT_TARGET hipfort::hipfort-nvptx)
    elseif(TARGET hipfort::hipfort)
      set(HIPFORT_TARGET hipfort::hipfort)
    endif()

    if(NOT HIPFORT_TARGET)
      # Fallback: look for hipfort manually in ROCM_PATH
      if(DEFINED ENV{ROCM_PATH})
        set(HIPFORT_ROOT "$ENV{ROCM_PATH}")
      elseif(EXISTS "/opt/rocm")
        set(HIPFORT_ROOT "/opt/rocm")
      endif()

      if(HIPFORT_ROOT)
        find_library(HIPFORT_LIB
          NAMES hipfort-amdgcn hipfort
          HINTS "${HIPFORT_ROOT}/lib" "${HIPFORT_ROOT}/lib64"
        )
        find_path(HIPFORT_INCLUDE_DIR
          NAMES hipfort.mod
          HINTS "${HIPFORT_ROOT}/include/hipfort/amdgcn"
                "${HIPFORT_ROOT}/include/hipfort"
        )
        if(HIPFORT_LIB AND HIPFORT_INCLUDE_DIR)
          set(HIPFORT_MANUAL TRUE)
        endif()
      endif()
    endif()

    if(HIPFORT_TARGET OR HIPFORT_MANUAL)
      message(STATUS "Building example_f03_hip with hipfort")

      add_executable(example_f03_hip example.f03)

      if(HIPFORT_TARGET)
        target_link_libraries(example_f03_hip PRIVATE ${HIPFORT_TARGET})
      else()
        target_include_directories(example_f03_hip PRIVATE ${HIPFORT_INCLUDE_DIR})
        target_link_libraries(example_f03_hip PRIVATE ${HIPFORT_LIB})
      endif()

      if(SHAFFT_EXAMPLES_STANDALONE)
        target_link_libraries(example_f03_hip
          PRIVATE
            shafft::shafftf03
            MPI::MPI_Fortran
            PkgConfig::FFTW3
            PkgConfig::FFTW3F
            ${FFTW_THREADS_TARGET}
        )
      else()
        target_include_directories(example_f03_hip PRIVATE
          ${CMAKE_BINARY_DIR}/include  # Location of shafft.mod
        )
        target_link_libraries(example_f03_hip
          PRIVATE
            shafftf03
            MPI::MPI_Fortran
        )
      endif()
    else()
      message(WARNING "SHAFFT_BUILD_HIPFORT_EXAMPLE=ON but hipfort not found - skipping example_f03_hip")
    endif()
  endif()
endif()
