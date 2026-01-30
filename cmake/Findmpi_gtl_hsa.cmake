# Findmpi_gtl_hsa.cmake
# Custom CMake module to find GTL libraries on Cray systems

# Check if the system is a Cray system
if(NOT DEFINED ENV{CRAY_CPU_TARGET})
    message(STATUS "Not a Cray system")
    set(mpi_gtl_FOUND FALSE)
    return()
endif()

# Check if there are any environment variables starting with PE_MPICH_GTL_DIR
execute_process(
    COMMAND env
    OUTPUT_VARIABLE ENV_VARS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

string(REGEX MATCHALL "PE_MPICH_GTL_DIR_[^=]+" GTL_DIR_VARS "${ENV_VARS}")

if(NOT GTL_DIR_VARS)
    message(STATUS "No GTL support detected")
    set(mpi_gtl_FOUND FALSE)
    return()
endif()

# Check HIP_PLATFORM for vendor
if(DEFINED ENV{HIP_PLATFORM})
    if("$ENV{HIP_PLATFORM}" STREQUAL "nvidia")
        if(DEFINED ENV{CUDA_VISIBLE_DEVICES})
            set(GTL_ARCH "nvidia80")  # Default to nvidia80 if CUDA_VISIBLE_DEVICES is set
        else()
            # If CUDA_VISIBLE_DEVICES is not defined, use the first matched PE_MPICH_GTL_DIR variable
            list(GET GTL_DIR_VARS 0 FIRST_GTL_DIR_VAR)
            string(REPLACE "PE_MPICH_GTL_DIR_" "" GTL_ARCH "${FIRST_GTL_DIR_VAR}")
        endif()
    elseif("$ENV{HIP_PLATFORM}" STREQUAL "amd")
        if(DEFINED ENV{HCC_AMDGPU_TARGET})
            set(GTL_ARCH "amd_${HCC_AMDGPU_TARGET}")
        else()
            # If HCC_AMDGPU_TARGET is not defined, use the first matched PE_MPICH_GTL_DIR variable
            list(GET GTL_DIR_VARS 0 FIRST_GTL_DIR_VAR)
            string(REPLACE "PE_MPICH_GTL_DIR_" "" GTL_ARCH "${FIRST_GTL_DIR_VAR}")
        endif()
    else()
        message(FATAL_ERROR "Unsupported HIP_PLATFORM: $ENV{HIP_PLATFORM}")
    endif()
else()
    # If HIP_PLATFORM is not defined, use the first matched PE_MPICH_GTL_DIR variable
    list(GET GTL_DIR_VARS 0 FIRST_GTL_DIR_VAR)
    string(REPLACE "PE_MPICH_GTL_DIR_" "" GTL_ARCH "${FIRST_GTL_DIR_VAR}")
endif()

# Extract GTL directory and library names
string(REPLACE "-L" "" GTL_DIR "$ENV{PE_MPICH_GTL_DIR_${GTL_ARCH}}")
string(REPLACE "-l" "" GTL_LIB "$ENV{PE_MPICH_GTL_LIBS_${GTL_ARCH}}")

# Find the GTL library
find_library(MPI_GTL_LIB 
            ${GTL_LIB}
            PATHS ${GTL_DIR}
            NO_DEFAULT_PATH)

if(MPI_GTL_LIB)
    message(STATUS "Found GTL library for ${GTL_ARCH} at: ${MPI_GTL_LIB}")
    set(mpi_gtl_FOUND TRUE)
    set(mpi_gtl_LIBRARIES ${MPI_GTL_LIB})
    set(mpi_gtl_TYPE ${GTL_ARCH})

    # Create IMPORTED target for cleaner usage
    if(NOT TARGET mpi_gtl::mpi_gtl)
        add_library(mpi_gtl::mpi_gtl UNKNOWN IMPORTED)
        set_target_properties(mpi_gtl::mpi_gtl PROPERTIES
            IMPORTED_LOCATION "${MPI_GTL_LIB}"
        )
    endif()
else()
    message(FATAL_ERROR "GTL library not found in ${GTL_DIR}")
endif()

mark_as_advanced(mpi_gtl_FOUND mpi_gtl_LIBRARIES mpi_gtl_TYPE)
