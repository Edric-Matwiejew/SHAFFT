# C API object (always)
add_library(_shafftc OBJECT shafftc.cpp)
target_include_directories(_shafftc PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../cpp
)
target_link_libraries(_shafftc PRIVATE shafft_options MPI::MPI_CXX)

# Add HIP include directories for HIPFFT backend
if(SHAFFT_ENABLE_HIPFFT)
  target_link_libraries(_shafftc PRIVATE hip::host)
endif()

# Objects common to all backends
set(SHAFFTC_OBJECTS
  $<TARGET_OBJECTS:_shafftc>
  $<TARGET_OBJECTS:shafft>
  $<TARGET_OBJECTS:fft_method>
  ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/shafft.cpp  # Include C++ API functions (allocBuffer, etc.)
)

# Backend-specific objects & link libraries
if(SHAFFT_ENABLE_HIPFFT)
  # HIP backend: normalize is already included in fft_method

  add_library(shafftc SHARED ${SHAFFTC_OBJECTS})

  target_include_directories(shafftc PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # for internal headers
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp  # for _shafft.hpp
  )

  # Link HIP + MPI + common options
  target_link_libraries(shafftc
    PUBLIC
      shafft_options
      MPI::MPI_CXX
    PRIVATE
      hip::host
      hip::hipfft
  )

  # GTL is only needed on Cray systems with GPU-aware MPICH
  if(SHAFFT_HAVE_GTL)
    target_link_libraries(shafftc PUBLIC ${mpi_gtl_LIBRARIES})
  endif()

elseif(SHAFFT_ENABLE_FFTW)
  # FFTW backend: add CPU normalizer and build the C API shared lib
  list(APPEND SHAFFTC_OBJECTS
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/fft/fftw_method/normalize.cpp
  )

  add_library(shafftc SHARED ${SHAFFTC_OBJECTS})

  # Link exactly the targets defined by the egpbos FindFFTW
  # Base (double + float)
  target_link_libraries(shafftc
    PUBLIC  shafft_options MPI::MPI_CXX
    PRIVATE FFTW::Double FFTW::Float OpenMP::OpenMP_CXX
  )
  # Threads: link both double and float threading libraries based on user's choice
  if(SHAFFT_FFTW_USE_OMP)
    target_link_libraries(shafftc PRIVATE FFTW::DoubleOpenMP FFTW::FloatOpenMP)
  else()
    target_link_libraries(shafftc PRIVATE FFTW::DoubleThreads FFTW::FloatThreads)
  endif()

  target_include_directories(shafftc PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # for internal headers
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp  # for _shafft.hpp
    ${FFTW_INCLUDE_DIRS}
  )

else()
  message(FATAL_ERROR "Internal error: backend selection flags are inconsistent.")
endif()

# Ensure the C shared lib builds after the core objects
add_dependencies(shafftc shafft)

# Set library version
set_target_properties(shafftc PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Create ALIAS target for use with FetchContent/add_subdirectory
add_library(shafft::shafftc ALIAS shafftc)
