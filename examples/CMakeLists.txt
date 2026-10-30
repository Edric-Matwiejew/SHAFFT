cmake_minimum_required(VERSION 3.28)
project(shafft_examples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(TARGET shafft::shafftc++)
  message(STATUS "Building examples as part of SHAFFT build")
  set(SHAFFT_EXAMPLES_STANDALONE FALSE)
else()
  message(STATUS "Building examples standalone - using find_package(shafft)")
  set(SHAFFT_EXAMPLES_STANDALONE TRUE)

  find_package(shafft CONFIG REQUIRED HINTS "${SHAFFT_PATH}" "${SHAFFT_ROOT}" "$ENV{SHAFFT_ROOT}")

  if(SHAFFT_BACKEND_HIPFFT)
    message(STATUS "SHAFFT was built with HIPFFT backend")
  else()
    message(STATUS "SHAFFT was built with FFTW backend")
  endif()
endif()

find_package(MPI REQUIRED)

if(SHAFFT_BACKEND_HIPFFT)
  add_executable(example_hip example.cpp)

  target_link_libraries(example_hip PRIVATE shafft::shafftc++ MPI::MPI_CXX)
endif()

add_executable(example_portable example_portable.cpp)

target_link_libraries(example_portable PRIVATE shafft::shafftc++ MPI::MPI_CXX)

add_executable(example_portable_config example_portable_config.cpp)

target_link_libraries(example_portable_config PRIVATE shafft::shafftc++ MPI::MPI_CXX)

add_executable(example_FFT1D example_FFT1D.cpp)

target_link_libraries(example_FFT1D PRIVATE shafft::shafftc++ MPI::MPI_CXX)

enable_language(C)

find_package(MPI COMPONENTS C REQUIRED)

add_executable(example_portable_c example_portable.c)

target_link_libraries(example_portable_c PRIVATE shafft::shafftc MPI::MPI_C m)

add_executable(example_portable_c_config example_portable_config.c)

target_link_libraries(example_portable_c_config PRIVATE shafft::shafftc MPI::MPI_C m)

if(SHAFFT_EXAMPLES_STANDALONE AND TARGET shafft::shafftf03)
  enable_language(Fortran)
  find_package(MPI COMPONENTS Fortran REQUIRED)

  add_executable(example_f03_portable example_portable.f03)

  add_executable(example_f03_portable_config example_portable_config.f03)

  target_link_libraries(
    example_f03_portable
    PRIVATE
      shafft::shafftf03
      MPI::MPI_Fortran
  )

  target_link_libraries(
    example_f03_portable_config
    PRIVATE
      shafft::shafftf03
      MPI::MPI_Fortran
  )
elseif(NOT SHAFFT_EXAMPLES_STANDALONE AND SHAFFT_BUILD_FORTRAN)
  enable_language(Fortran)
  find_package(MPI COMPONENTS Fortran REQUIRED)

  add_executable(example_f03_portable example_portable.f03)
  add_executable(example_f03_portable_config example_portable_config.f03)

  set_target_properties(example_f03_portable PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE)
  target_compile_options(example_f03_portable PRIVATE $<$<Fortran_COMPILER_ID:GNU>:-fno-lto>)
  target_link_options(example_f03_portable PRIVATE $<$<Fortran_COMPILER_ID:GNU>:-fno-lto>)
  set_target_properties(example_f03_portable_config PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE)
  target_compile_options(example_f03_portable_config PRIVATE $<$<Fortran_COMPILER_ID:GNU>:-fno-lto>)
  target_link_options(example_f03_portable_config PRIVATE $<$<Fortran_COMPILER_ID:GNU>:-fno-lto>)

  target_include_directories(
    example_f03_portable
    PRIVATE
      ${CMAKE_BINARY_DIR}/include
  )
  target_include_directories(
    example_f03_portable_config
    PRIVATE
      ${CMAKE_BINARY_DIR}/include
  )

  target_link_libraries(
    example_f03_portable
    PRIVATE
      shafftf03
      MPI::MPI_Fortran
  )
  target_link_libraries(
    example_f03_portable_config
    PRIVATE
      shafftf03
      MPI::MPI_Fortran
  )

  add_dependencies(example_f03_portable shafftf03)
  add_dependencies(example_f03_portable_config shafftf03)
endif()

option(
  SHAFFT_BUILD_HIPFORT_EXAMPLE
  "Build Fortran example using hipfort (requires compatible hipfort)"
  OFF
)

if(SHAFFT_BUILD_HIPFORT_EXAMPLE)
  if(NOT SHAFFT_BACKEND_HIPFFT)
    message(
      WARNING
      "SHAFFT_BUILD_HIPFORT_EXAMPLE=ON but HIPFFT backend not enabled - skipping example_f03_hip"
    )
  else()
    enable_language(Fortran)
    find_package(MPI COMPONENTS Fortran REQUIRED)

    if(SHAFFT_EXAMPLES_STANDALONE)
      list(APPEND CMAKE_MODULE_PATH "${SHAFFT_PATH}/share/cmake/Modules")
    else()
      list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
    endif()

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
        )
      else()
        target_include_directories(
          example_f03_hip
          PRIVATE
            ${CMAKE_BINARY_DIR}/include
        )
        target_link_libraries(example_f03_hip PRIVATE shafftf03 MPI::MPI_Fortran)
      endif()
    else()
      message(WARNING "hipfort not found; skip example_f03_hip")
    endif()
  endif()
endif()
