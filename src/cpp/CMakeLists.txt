# src/cpp/CMakeLists.txt

# ---------------- HIPFFT backend ----------------
if(SHAFFT_ENABLE_HIPFFT)

  add_library(fft_method OBJECT
    fft/hipfft_method/fft_method.cpp
    fft/hipfft_method/transpose_fft.cpp
    fft/hipfft_method/normalize.hip
  )

  set(FFT_METHOD_PATH
      ${CMAKE_CURRENT_SOURCE_DIR}/fft/hipfft_method
      CACHE PATH "Path to fft method")

  target_include_directories(fft_method PRIVATE
    ${PROJECT_BINARY_DIR}/include
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${FFT_METHOD_PATH}
  )
  target_link_libraries(fft_method PRIVATE shafft_options hip::host hip::hipfft MPI::MPI_CXX gputt)

  add_library(shafft OBJECT
    slab.cpp
    comm_dims.cpp
    configuration.cpp
    _shafft.cpp
    shafft_error.cpp
  )
  target_include_directories(shafft PRIVATE
    ${PROJECT_BINARY_DIR}/include
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${FFT_METHOD_PATH}
  )
  target_link_libraries(shafft PRIVATE shafft_options MPI::MPI_CXX hip::host hip::hipfft gputt)
  if(SHAFFT_HAVE_GTL)
    target_link_libraries(shafft PRIVATE CrayGTL::CrayGTL)
  endif()

  add_library(shafftc++ SHARED
    shafft.cpp
    $<TARGET_OBJECTS:shafft>
    $<TARGET_OBJECTS:fft_method>
  )
  target_include_directories(shafftc++ PUBLIC
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  target_include_directories(shafftc++ PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(shafftc++ PUBLIC shafft_options MPI::MPI_CXX PRIVATE hip::host hip::hipfft gputt)
  if(SHAFFT_HAVE_GTL)
    target_link_libraries(shafftc++ PRIVATE CrayGTL::CrayGTL)
  endif()
  add_dependencies(shafftc++ shafft)

# ---------------- FFTW backend ----------------
elseif(SHAFFT_ENABLE_FFTW)

  if(SHAFFT_FFTW_USE_OMP)
    find_package(OpenMP REQUIRED)
  endif()

  add_library(fft_method OBJECT
    fft/fftw_method/fft_method.cpp
  )
  target_include_directories(fft_method PRIVATE
    ${PROJECT_BINARY_DIR}/include
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR} ${FFTW_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/fft/fftw_method
  )
  target_link_libraries(fft_method PRIVATE shafft_options MPI::MPI_CXX)

  add_library(shafft OBJECT
    slab.cpp
    comm_dims.cpp
    configuration.cpp
    _shafft.cpp
    shafft_error.cpp
  )
  target_include_directories(shafft PRIVATE
    ${PROJECT_BINARY_DIR}/include
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/fft/fftw_method
    ${FFTW_INCLUDE_DIRS}
  )
  target_link_libraries(shafft PRIVATE shafft_options MPI::MPI_CXX)

  # FFTW normalize (OpenMP)
  set(FFT_BACKEND_CPU_EXTRAS
    fft/fftw_method/normalize.cpp
  )

  add_library(shafftc++ SHARED
  shafft.cpp
  ${FFT_BACKEND_CPU_EXTRAS}
  $<TARGET_OBJECTS:shafft>
  $<TARGET_OBJECTS:fft_method>
)

# Base FFTW (double + float) - required
target_link_libraries(shafftc++
  PUBLIC  MPI::MPI_CXX
  PRIVATE FFTW::Double FFTW::Float
)

# Threads: link both double and float threading libraries based on user's choice
if(SHAFFT_FFTW_USE_OMP)
  target_link_libraries(shafftc++ PRIVATE FFTW::DoubleOpenMP FFTW::FloatOpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(shafftc++ PRIVATE OpenMP::OpenMP_CXX)
  endif()
else()
  target_link_libraries(shafftc++ PRIVATE FFTW::DoubleThreads FFTW::FloatThreads)
endif()

target_include_directories(shafftc++ 
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/fft/fftw_method
    ${FFTW_INCLUDE_DIRS}
)

target_link_libraries(shafftc++ PUBLIC shafft_options)

add_dependencies(shafftc++ shafft)

else()
  message(FATAL_ERROR "Internal error: backend selection flags are inconsistent.")
endif()

set_target_properties(shafftc++ PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  CXX_VISIBILITY_PRESET default
  VISIBILITY_INLINES_HIDDEN OFF
)

# Create ALIAS target for use with FetchContent/add_subdirectory
add_library(shafft::shafftc++ ALIAS shafftc++)
