cmake_minimum_required(VERSION 3.21)
project(shafft_examples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(TARGET shafft::shafftc++)
  message(STATUS "Building examples as part of SHAFFT build")
  set(SHAFFT_EXAMPLES_STANDALONE FALSE)
else()
  message(STATUS "Building examples standalone - using find_package(shafft)")
  set(SHAFFT_EXAMPLES_STANDALONE TRUE)

  if(NOT DEFINED SHAFFT_PATH)
    message(FATAL_ERROR "Please configure with -DSHAFFT_PATH=/path/to/shafft/install/prefix")
  endif()

  find_package(
    shafft
    REQUIRED
    HINTS
      "${SHAFFT_PATH}/lib/cmake/shafft"
      "${SHAFFT_PATH}/${CMAKE_INSTALL_LIBDIR}/cmake/shafft"
      "${SHAFFT_PATH}/lib64/cmake/shafft"
  )

  if(SHAFFT_BACKEND_HIPFFT)
    set(SHAFFT_USE_HIPFFT TRUE)
    message(STATUS "SHAFFT was built with HIPFFT backend")
  else()
    set(SHAFFT_USE_HIPFFT FALSE)
    message(STATUS "SHAFFT was built with FFTW backend")
  endif()
endif()

if(SHAFFT_EXAMPLES_STANDALONE)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(FFTW3 REQUIRED IMPORTED_TARGET fftw3)
  pkg_check_modules(FFTW3F REQUIRED IMPORTED_TARGET fftw3f)

  set(FFTW_THREADS_TARGET "")
  pkg_check_modules(FFTW3_THREADS QUIET IMPORTED_TARGET fftw3_threads)
  if(FFTW3_THREADS_FOUND)
    set(FFTW_THREADS_TARGET PkgConfig::FFTW3_THREADS)
  else()
    pkg_check_modules(FFTW3_OMP QUIET IMPORTED_TARGET fftw3_omp)
    if(FFTW3_OMP_FOUND)
      set(FFTW_THREADS_TARGET PkgConfig::FFTW3_OMP)
    else()
      find_library(
        FFTW_THREADS_LIB
        NAMES fftw3_threads fftw3_omp
        HINTS /opt/homebrew/lib
        ENV LIBRARY_PATH
        ${CMAKE_PREFIX_PATH}
      )
      if(NOT FFTW_THREADS_LIB)
        message(FATAL_ERROR "Could not find FFTW threads library (fftw3_threads or fftw3_omp).")
      endif()
      set(FFTW_THREADS_TARGET "${FFTW_THREADS_LIB}")
    endif()
  endif()
endif()

find_package(MPI REQUIRED)

if(SHAFFT_USE_HIPFFT OR SHAFFT_ENABLE_HIPFFT)
  add_executable(example_hip example.cpp)

  target_link_libraries(example_hip PRIVATE shafft::shafftc++ MPI::MPI_CXX)

  if(SHAFFT_EXAMPLES_STANDALONE)
    target_link_libraries(
      example_hip
      PRIVATE PkgConfig::FFTW3 PkgConfig::FFTW3F ${FFTW_THREADS_TARGET}
    )
  endif()
endif()

add_executable(example_portable example_portable.cpp)

target_link_libraries(example_portable PRIVATE shafft::shafftc++ MPI::MPI_CXX)

if(SHAFFT_EXAMPLES_STANDALONE)
  target_link_libraries(
    example_portable
    PRIVATE PkgConfig::FFTW3 PkgConfig::FFTW3F ${FFTW_THREADS_TARGET}
  )
endif()

enable_language(C)

find_package(MPI COMPONENTS C REQUIRED)

add_executable(example_c example.c)

target_link_libraries(example_c PRIVATE shafft::shafftc MPI::MPI_C m)

if(SHAFFT_EXAMPLES_STANDALONE)
  target_link_libraries(example_c PRIVATE PkgConfig::FFTW3 PkgConfig::FFTW3F ${FFTW_THREADS_TARGET})
endif()

if(SHAFFT_EXAMPLES_STANDALONE AND TARGET shafft::shafftf03)
  enable_language(Fortran)
  find_package(MPI COMPONENTS Fortran REQUIRED)

  add_executable(example_f03_portable example_portable.f03)

  target_link_libraries(
    example_f03_portable
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

  # Disable LTO for this target (hipfort may set -flto in CMAKE_Fortran_FLAGS
  # which can cause issues with MPI F08 symbol resolution)
  set_target_properties(example_f03_portable PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE)
  target_compile_options(example_f03_portable PRIVATE $<$<Fortran_COMPILER_ID:GNU>:-fno-lto>)
  target_link_options(example_f03_portable PRIVATE $<$<Fortran_COMPILER_ID:GNU>:-fno-lto>)

  target_include_directories(
    example_f03_portable
    PRIVATE
      ${CMAKE_BINARY_DIR}/include # Location of shafft.mod
  )

  target_link_libraries(
    example_f03_portable
    PRIVATE
      shafftf03 # Real target name (not alias)
      MPI::MPI_Fortran
  )

  # Ensure .mod files are built before compiling this example
  add_dependencies(example_f03_portable shafftf03)
endif()

option(
  SHAFFT_BUILD_HIPFORT_EXAMPLE
  "Build Fortran example using hipfort (requires compatible hipfort)"
  OFF
)

if(SHAFFT_BUILD_HIPFORT_EXAMPLE)
  if(NOT (SHAFFT_USE_HIPFFT OR SHAFFT_ENABLE_HIPFFT))
    message(
      WARNING
      "SHAFFT_BUILD_HIPFORT_EXAMPLE=ON but HIPFFT backend not enabled - skipping example_f03_hip"
    )
  else()
    enable_language(Fortran)
    find_package(MPI COMPONENTS Fortran REQUIRED)

    # Add cmake module path for HipfortDependency
    if(SHAFFT_EXAMPLES_STANDALONE)
      list(APPEND CMAKE_MODULE_PATH "${SHAFFT_PATH}/share/cmake/Modules")
    else()
      list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
    endif()

    # Use HipfortDependency module for clean hipfort detection
    include(HipfortDependency)

    if(TARGET hipfort::hipfort)
      message(STATUS "Building example_f03_hip with hipfort (${HIPFORT_PROVIDER})")

      add_executable(example_f03_hip example.f03)

      target_link_libraries(example_f03_hip PRIVATE hipfort::hipfort)

      if(SHAFFT_EXAMPLES_STANDALONE)
        target_link_libraries(
          example_f03_hip
          PRIVATE
            shafft::shafftf03
            MPI::MPI_Fortran
            PkgConfig::FFTW3
            PkgConfig::FFTW3F
            ${FFTW_THREADS_TARGET}
        )
      else()
        target_include_directories(
          example_f03_hip
          PRIVATE
            ${CMAKE_BINARY_DIR}/include # Location of shafft.mod
        )
        target_link_libraries(example_f03_hip PRIVATE shafftf03 MPI::MPI_Fortran)
      endif()
    else()
      message(WARNING "hipfort not found; skip example_f03_hip")
    endif()
  endif()
endif()
