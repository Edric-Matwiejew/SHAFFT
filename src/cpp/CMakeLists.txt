if(SHAFFT_ENABLE_HIPFFT)

  add_library(fftnd_method OBJECT
    fft/hipfft_method/fftnd_method.cpp
    fft/hipfft_method/nd/transpose_fftnd.cpp
    fft/hipfft_method/common/normalize.hip
  )

  add_library(fft1d_method OBJECT
    fft/hipfft_method/fft1d_method.hip
  )

  set(FFT_METHOD_PATH
      ${CMAKE_CURRENT_SOURCE_DIR}/fft/hipfft_method
      CACHE PATH "Path to FFT method implementations")

  target_include_directories(fftnd_method PRIVATE
    ${PROJECT_BINARY_DIR}/include
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${FFT_METHOD_PATH}
  )
  target_link_libraries(
    fftnd_method PRIVATE shafft_options hip::host hip::hipfft MPI::MPI_CXX gputt
  )

  target_include_directories(fft1d_method PRIVATE
    ${PROJECT_BINARY_DIR}/include
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${FFT_METHOD_PATH}
  )
  target_link_libraries(fft1d_method PRIVATE shafft_options hip::host hip::hipfft MPI::MPI_CXX)

  add_library(shafft OBJECT
    slab_nd.cpp
    configuration.cpp
    config_info.cpp
    shafft_internal.cpp
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
    $<TARGET_OBJECTS:fftnd_method>
    $<TARGET_OBJECTS:fft1d_method>
  )
  target_include_directories(shafftc++ PUBLIC
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  target_include_directories(shafftc++ PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${FFT_METHOD_PATH}
  )
  target_link_libraries(
    shafftc++ PUBLIC shafft_options MPI::MPI_CXX PRIVATE hip::host hip::hipfft gputt
  )
  if(SHAFFT_HAVE_GTL)
    target_link_libraries(shafftc++ PRIVATE CrayGTL::CrayGTL)
  endif()
  add_dependencies(shafftc++ shafft fft1d_method)

elseif(SHAFFT_ENABLE_FFTW)

  if(SHAFFT_FFTW_USE_OMP)
    find_package(OpenMP REQUIRED)
  endif()

  set(FFT_METHOD_PATH
      ${CMAKE_CURRENT_SOURCE_DIR}/fft/fftw_method
      CACHE PATH "Path to FFT method implementations")

  add_library(fftnd_method OBJECT
    fft/fftw_method/fftnd_method.cpp
  )
  target_include_directories(fftnd_method PRIVATE
    ${PROJECT_BINARY_DIR}/include
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR} ${FFTW_INCLUDE_DIRS}
    ${FFT_METHOD_PATH}
  )
  target_link_libraries(fftnd_method PRIVATE shafft_options MPI::MPI_CXX)

  add_library(fft1d_method OBJECT
    fft/fftw_method/fft1d_method.cpp
  )
  target_include_directories(fft1d_method PRIVATE
    ${PROJECT_BINARY_DIR}/include
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${FFT_METHOD_PATH}
    ${FFTW_INCLUDE_DIRS}
  )
  target_link_libraries(fft1d_method PRIVATE shafft_options MPI::MPI_CXX)

  add_library(shafft OBJECT
    slab_nd.cpp
    configuration.cpp
    config_info.cpp
    shafft_internal.cpp
    shafft_error.cpp
  )
  target_include_directories(shafft PRIVATE
    ${PROJECT_BINARY_DIR}/include
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${FFT_METHOD_PATH}
    ${FFTW_INCLUDE_DIRS}
  )
  target_link_libraries(shafft PRIVATE shafft_options MPI::MPI_CXX)

  set(FFT_BACKEND_CPU_EXTRAS
    fft/fftw_method/common/normalize.cpp
  )

  add_library(shafftc++ SHARED
  shafft.cpp
  ${FFT_BACKEND_CPU_EXTRAS}
  $<TARGET_OBJECTS:shafft>
  $<TARGET_OBJECTS:fftnd_method>
  $<TARGET_OBJECTS:fft1d_method>
)

target_link_libraries(shafftc++
  PUBLIC  MPI::MPI_CXX
  PRIVATE FFTW::Double FFTW::Float FFTW::DoubleMPI FFTW::FloatMPI
)

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
    $<BUILD_INTERFACE:${FFTW_INCLUDE_DIRS}>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${FFT_METHOD_PATH}
)

target_link_libraries(shafftc++ PUBLIC shafft_options)

add_dependencies(shafftc++ shafft fft1d_method)

else()
  message(FATAL_ERROR "Internal error: backend selection flags are inconsistent.")
endif()

set_target_properties(shafftc++ PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  CXX_VISIBILITY_PRESET default
  VISIBILITY_INLINES_HIDDEN OFF
)

add_library(shafft::shafftc++ ALIAS shafftc++)
