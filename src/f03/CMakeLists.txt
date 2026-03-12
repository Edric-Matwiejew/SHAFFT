enable_language(Fortran)

set(CMAKE_Fortran_PREPROCESS ON)

set(SHAFFTF03_OBJECTS
  shafft.f03 
  shafftf03.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/shafft.cpp
  $<TARGET_OBJECTS:_shafftc>
  $<TARGET_OBJECTS:shafft>
  $<TARGET_OBJECTS:fftnd_method>
  $<TARGET_OBJECTS:fft1d_method>
)

if(SHAFFT_ENABLE_FFTW)
  list(APPEND SHAFFTF03_OBJECTS
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/fft/fftw_method/common/normalize.cpp
  )
endif()

add_library(shafftf03 SHARED ${SHAFFTF03_OBJECTS})

set_target_properties(shafftf03 PROPERTIES LINKER_LANGUAGE Fortran)

add_dependencies(shafftf03 shafft fftnd_method fft1d_method)

target_link_libraries(shafftf03 PRIVATE shafft_options)

target_include_directories(shafftf03
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_Fortran_MODULE_DIRECTORY}>
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/shafft>
  PRIVATE	
    ${CMAKE_CURRENT_SOURCE_DIR}/../c
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp
    ${FFT_METHOD_PATH}
    ${PROJECT_BINARY_DIR}/include/shafft
)

if(SHAFFT_ENABLE_HIPFFT)
  target_link_libraries(shafftf03 
    PRIVATE hip::host hip::hipfft gputt
    PUBLIC MPI::MPI_CXX
  )
  if(SHAFFT_HAVE_GTL)
    target_link_libraries(shafftf03 PRIVATE CrayGTL::CrayGTL)
  endif()
elseif(SHAFFT_ENABLE_FFTW)
  target_link_libraries(shafftf03 
    PUBLIC MPI::MPI_CXX
    PRIVATE FFTW::Double FFTW::Float FFTW::DoubleMPI FFTW::FloatMPI
  )
  if(SHAFFT_FFTW_USE_OMP)
    target_link_libraries(shafftf03 PRIVATE FFTW::DoubleOpenMP FFTW::FloatOpenMP)
    if(OpenMP_CXX_FOUND)
      target_link_libraries(shafftf03 PRIVATE OpenMP::OpenMP_CXX)
    endif()
  else()
    target_link_libraries(shafftf03 PRIVATE FFTW::DoubleThreads FFTW::FloatThreads)
  endif()
  target_include_directories(shafftf03 PRIVATE ${FFTW_INCLUDE_DIRS})
endif()
