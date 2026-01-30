enable_language(Fortran)

set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/include)
set(CMAKE_Fortran_PREPROCESS ON)

# Collect all objects needed for a self-contained Fortran library
set(SHAFFTF03_OBJECTS
  shafft.f03 
  shafftf03.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/shafft.cpp  # C++ API functions (allocBuffer, etc.)
  $<TARGET_OBJECTS:_shafftc>
  $<TARGET_OBJECTS:shafft>
  $<TARGET_OBJECTS:fft_method>
)

# Add FFTW-specific objects if needed
if(SHAFFT_ENABLE_FFTW)
  list(APPEND SHAFFTF03_OBJECTS
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/fft/fftw_method/normalize.cpp
  )
endif()

# Build the self-contained Fortran library
add_library(shafftf03 SHARED ${SHAFFTF03_OBJECTS})

add_dependencies(shafftf03 shafft fft_method)

# Link to shafft_options for public header include paths and backend dependencies
target_link_libraries(shafftf03 PRIVATE shafft_options)

target_include_directories(shafftf03
  PRIVATE	
  ${CMAKE_CURRENT_SOURCE_DIR}/../c
  ${CMAKE_CURRENT_SOURCE_DIR}/../cpp
  # Fortran preprocessor uses #include "shafft_config.h" (no prefix)
  ${PROJECT_BINARY_DIR}/include/shafft
)

# Link backend-specific libraries
if(SHAFFT_ENABLE_HIPFFT)
  target_link_libraries(shafftf03 
    PRIVATE hip::host hip::hipfft ${mpi_gtl_LIBRARIES}
    PUBLIC MPI::MPI_CXX
  )
elseif(SHAFFT_ENABLE_FFTW)
  target_link_libraries(shafftf03 
    PUBLIC MPI::MPI_CXX
    PRIVATE FFTW::Double FFTW::Float OpenMP::OpenMP_CXX
  )
  if(SHAFFT_FFTW_USE_OMP)
    target_link_libraries(shafftf03 PRIVATE FFTW::DoubleOpenMP FFTW::FloatOpenMP)
  else()
    target_link_libraries(shafftf03 PRIVATE FFTW::DoubleThreads FFTW::FloatThreads)
  endif()
  target_include_directories(shafftf03 PRIVATE ${FFTW_INCLUDE_DIRS})
endif()
