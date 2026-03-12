#!/bin/bash
# Test script for FindCrayGTL.cmake on Cray systems
# Run this from the cmake/test-craygtl directory

set -e

echo "=== FindCrayGTL Test ==="
echo ""

# Show relevant environment
echo "Environment:"
echo "  CRAY_CPU_TARGET=${CRAY_CPU_TARGET:-<not set>}"
echo "  CRAY_MPICH_DIR=${CRAY_MPICH_DIR:-<not set>}"
echo ""

# Show available GTL variables
echo "GTL Environment Variables:"
env | grep -E "^PE_MPICH_GTL" | sort || echo "  (none found)"
echo ""

# Create build directory
rm -rf build
mkdir -p build
cd build

# Configure with CMake
echo "Running CMake..."
echo ""

# If HIP architectures should be set, pass them
if [[ -n "${ROCM_GPU}" ]]; then
    cmake .. -DCMAKE_HIP_ARCHITECTURES="${ROCM_GPU}"
elif [[ -n "${CRAY_ACCEL_TARGET}" ]]; then
    # On some Cray systems, CRAY_ACCEL_TARGET contains the GPU arch
    case "${CRAY_ACCEL_TARGET}" in
        amd_gfx90a|gfx90a) cmake .. -DCMAKE_HIP_ARCHITECTURES=gfx90a ;;
        amd_gfx908|gfx908) cmake .. -DCMAKE_HIP_ARCHITECTURES=gfx908 ;;
        nvidia80) cmake .. -DCMAKE_CUDA_ARCHITECTURES=80 ;;
        nvidia90) cmake .. -DCMAKE_CUDA_ARCHITECTURES=90 ;;
        *) cmake .. ;;
    esac
else
    cmake ..
fi

echo ""
echo "=== Done ==="
