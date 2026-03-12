add_library(_shafftc OBJECT shafftc.cpp)
target_include_directories(_shafftc PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../cpp
)
target_link_libraries(_shafftc PRIVATE shafft_options MPI::MPI_CXX)

if(SHAFFT_ENABLE_HIPFFT)
  target_link_libraries(_shafftc PRIVATE hip::host)
endif()

if(SHAFFT_ENABLE_FFTW)
  target_include_directories(_shafftc PRIVATE ${FFTW_INCLUDE_DIRS})
endif()

set(SHAFFTC_OBJECTS
  $<TARGET_OBJECTS:_shafftc>
  $<TARGET_OBJECTS:shafft>
  $<TARGET_OBJECTS:fftnd_method>
  $<TARGET_OBJECTS:fft1d_method>
  ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/shafft.cpp
)

if(SHAFFT_ENABLE_HIPFFT)
  add_library(shafftc SHARED ${SHAFFTC_OBJECTS})

  target_include_directories(shafftc
    PUBLIC
      $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/../cpp
      ${FFT_METHOD_PATH}
  )

  target_link_libraries(shafftc
    PUBLIC
      shafft_options
      MPI::MPI_CXX
    PRIVATE
      hip::host
      hip::hipfft
      gputt
  )

  if(SHAFFT_HAVE_GTL)
    target_link_libraries(shafftc PRIVATE CrayGTL::CrayGTL)
  endif()

elseif(SHAFFT_ENABLE_FFTW)
  list(APPEND SHAFFTC_OBJECTS
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/fft/fftw_method/normalize.cpp
  )

  add_library(shafftc SHARED ${SHAFFTC_OBJECTS})

  target_link_libraries(shafftc
    PUBLIC  shafft_options MPI::MPI_CXX
    PRIVATE FFTW::Double FFTW::Float FFTW::DoubleMPI FFTW::FloatMPI
  )
  if(SHAFFT_FFTW_USE_OMP)
    target_link_libraries(shafftc PRIVATE FFTW::DoubleOpenMP FFTW::FloatOpenMP)
    if(OpenMP_CXX_FOUND)
      target_link_libraries(shafftc PRIVATE OpenMP::OpenMP_CXX)
    endif()
  else()
    target_link_libraries(shafftc PRIVATE FFTW::DoubleThreads FFTW::FloatThreads)
  endif()

  target_include_directories(shafftc
    PUBLIC
      $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/../cpp
      ${FFT_METHOD_PATH}
      ${FFTW_INCLUDE_DIRS}
  )

else()
  message(FATAL_ERROR "Internal error: backend selection flags are inconsistent.")
endif()

add_dependencies(shafftc shafft)

set_target_properties(shafftc PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
)

add_library(shafft::shafftc ALIAS shafftc)
