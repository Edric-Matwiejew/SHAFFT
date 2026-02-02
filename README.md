# SHAFFT

[![Tests](https://github.com/Edric-Matwiejew/SHAFFT/actions/workflows/fftw-tests.yml/badge.svg)](https://github.com/Edric-Matwiejew/SHAFFT/actions/workflows/fftw-tests.yml)
[![Docs](https://github.com/Edric-Matwiejew/SHAFFT/actions/workflows/build_doxygen.yml/badge.svg)](https://github.com/Edric-Matwiejew/SHAFFT/actions/workflows/build_doxygen.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

**SHAFFT** is a scalable library for high-dimensional complex-to-complex Fast Fourier Transforms (FFTs) in distributed-memory environments. It implements the slab decomposition method introduced by Dalcin, Mortensen, and Keyes ([arXiv:1804.09536](https://arxiv.org/abs/1804.09536)), using MPI for communication across a Cartesian process topology.

## Features

- N-dimensional distributed FFTs with slab decomposition
- Flexible process grid topology (1 to N-1 distributed axes)
- Single and double precision complex transforms
- Backend-agnostic design with portable buffer API
- C++, C, and Fortran 2003 interfaces

## Supported Backends

| Backend | Target | Description |
|---------|--------|-------------|
| **hipFFT** | GPU | AMD and NVIDIA GPUs via ROCm/HIP |
| **FFTW** | CPU | Multi-threaded CPU execution |

## Quick Start

### Requirements

- CMake >= 3.21
- MPI implementation
- GCC >= 10 (or C++17-compatible compiler)
- Backend: FFTW3 (CPU) or ROCm/HIP (GPU)

### Build

```bash
# FFTW backend (CPU)
cmake -B build -S . \
    -DSHAFFT_ENABLE_FFTW=ON \
    -DCMAKE_INSTALL_PREFIX=/opt/shafft

cmake --build build --target install
```

```bash
# hipFFT backend (GPU, host-staging MPI)
cmake -B build -S . \
    -DSHAFFT_ENABLE_HIPFFT=ON \
    -DSHAFFT_GPU_AWARE_MPI=OFF \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DCMAKE_INSTALL_PREFIX=/opt/shafft

cmake --build build --target install
```

Config header is generated at `build/include/shafft/shafft_config.h`; there should be no `include/shafft/shafft_config.h` in the source tree.

### Testing

```bash
cd build
ctest --output-on-failure

# Run specific test
ctest -R test_roundtrip_4ranks
```

### Example

```cpp
#include <shafft/shafft.hpp>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    shafft::Plan plan;
    plan.init(1, {64, 64, 32}, shafft::FFTType::C2C, MPI_COMM_WORLD);
    
    size_t n = plan.allocSize();
    shafft::complexf *data, *work;
    shafft::allocBuffer(n, &data);
    shafft::allocBuffer(n, &work);
    
    plan.setBuffers(data, work);
    plan.execute(shafft::FFTDirection::FORWARD);
    
    plan.release();
    shafft::freeBuffer(data);
    shafft::freeBuffer(work);
    
    MPI_Finalize();
    return 0;
}
```

## Documentation

For detailed documentation, see the generated Doxygen output:

```bash
doxygen docs/Doxyfile
open docs/html/index.html
```

Main documentation pages:
- @ref getting-started - Installation and build options
- @ref user-guide - API usage with C++, C, and Fortran examples
- @ref linking-guide - Compile and link instructions
- @ref backends - Backend-specific configuration
- @ref limitations - Known constraints and limitations

Or view the source documentation files directly in `docs/`.


## License

MIT License. See [LICENSE](./LICENSE) for details.

## Citation

If you use SHAFFT in your research, please cite the underlying method:

```bibtex
@article{dalcin2018fast,
  title={Fast parallel multidimensional FFT using advanced MPI},
  author={Dalcin, Lisandro and Mortensen, Mikael and Keyes, David E},
  journal={Journal of Parallel and Distributed Computing},
  volume={128},
  pages={137--150},
  year={2019},
  doi={10.1016/j.jpdc.2019.02.006}
}
```
