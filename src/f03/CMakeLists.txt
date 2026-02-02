enable_language(Fortran)

# CMAKE_Fortran_MODULE_DIRECTORY is set in parent CMakeLists.txt
set(CMAKE_Fortran_PREPROCESS ON)

set(SHAFFTF03_OBJECTS
  shafft.f03 
  shafftf03.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/shafft.cpp  # C++ API functions (allocBuffer, etc.)
  $<TARGET_OBJECTS:_shafftc>
  $<TARGET_OBJECTS:shafft>
  $<TARGET_OBJECTS:fft_method>
)

if(SHAFFT_ENABLE_FFTW)
  list(APPEND SHAFFTF03_OBJECTS
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/fft/fftw_method/normalize.cpp
  )
endif()

add_library(shafftf03 SHARED ${SHAFFTF03_OBJECTS})

# Force Fortran linker to ensure Fortran module symbols are preserved
# (HIP/C++ linker would strip them)
set_target_properties(shafftf03 PROPERTIES LINKER_LANGUAGE Fortran)

add_dependencies(shafftf03 shafft fft_method)

target_link_libraries(shafftf03 PRIVATE shafft_options)

target_include_directories(shafftf03
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE	
    ${CMAKE_CURRENT_SOURCE_DIR}/../c
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp
    ${PROJECT_BINARY_DIR}/include/shafft
)

if(SHAFFT_ENABLE_HIPFFT)
  target_link_libraries(shafftf03 
    PRIVATE hip::host hip::hipfft gputt $<TARGET_NAME_IF_EXISTS:CrayGTL::CrayGTL>
    PUBLIC MPI::MPI_CXX
  )
elseif(SHAFFT_ENABLE_FFTW)
  target_link_libraries(shafftf03 
    PUBLIC MPI::MPI_CXX
    PRIVATE FFTW::Double FFTW::Float
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
