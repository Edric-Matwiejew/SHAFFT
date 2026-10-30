# Some HIP releases leave hip::host empty on NVIDIA; complete its usage requirements.
include_guard(DIRECTORY)
find_package(CUDAToolkit REQUIRED)
find_path(SHAFFT_HIP_INCLUDE_DIR hip/hip_runtime.h
  HINTS ${hip_INCLUDE_DIRS} "${hip_DIR}/../../.." "${CMAKE_HIP_COMPILER_ROCM_ROOT}"
  ENV HIP_PATH ENV ROCM_PATH
  PATH_SUFFIXES include
  REQUIRED)
target_include_directories(hip::host INTERFACE "${SHAFFT_HIP_INCLUDE_DIR}")
target_compile_definitions(hip::host INTERFACE __HIP_PLATFORM_NVIDIA__)
target_link_libraries(hip::host INTERFACE CUDA::cudart)
