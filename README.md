# SHAFFT

[![Tests](https://github.com/Edric-Matwiejew/SHAFFT/actions/workflows/fftw-tests.yml/badge.svg)](https://github.com/Edric-Matwiejew/SHAFFT/actions/workflows/fftw-tests.yml)
[![Docs](https://github.com/Edric-Matwiejew/SHAFFT/actions/workflows/build_doxygen.yml/badge.svg)](https://github.com/Edric-Matwiejew/SHAFFT/actions/workflows/build_doxygen.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

**SHAFFT** is a scalable library for high-dimensional complex-to-complex Fast Fourier Transforms (FFTs) in distributed-memory environments. It implements the slab decomposition method introduced by Dalcin, Mortensen, and Keyes ([arXiv:1804.09536](https://arxiv.org/abs/1804.09536)), using MPI for communication across a Cartesian process topology.

## Features

- N-dimensional distributed FFTs with slab decomposition
- 1D distributed FFTs via dedicated FFT1D class
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
- C++17-compatible compiler (GCC >= 10)
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
# hipFFT backend (GPU)
cmake -B build -S . \
    -DSHAFFT_ENABLE_HIPFFT=ON \
    -DSHAFFT_GPU_AWARE_MPI=OFF \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DCMAKE_INSTALL_PREFIX=/opt/shafft

cmake --build build --target install
```

### Example

```cpp
#include <shafft/shafft.hpp>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    std::vector<int> commDims = {0, 0, 0};  // auto-select
    std::vector<size_t> dims = {64, 64, 32};
    
    shafft::FFTND plan;
    plan.init(commDims, dims, shafft::FFTType::C2C, MPI_COMM_WORLD);
    plan.plan();
    
    size_t n = plan.allocSize();
    shafft::complexf *data, *work;
    shafft::allocBuffer(n, &data);
    shafft::allocBuffer(n, &work);
    
    plan.setBuffers(data, work);
    plan.execute(shafft::FFTDirection::FORWARD);
    plan.normalize();
    
    plan.execute(shafft::FFTDirection::BACKWARD);
    plan.normalize();
    
    plan.release();
    shafft::freeBuffer(data);
    shafft::freeBuffer(work);
    
    MPI_Finalize();
    return 0;
}
```

### Validation (Optional)

```bash
cd build
ctest --output-on-failure
```

## Documentation

Primary documentation:
- [Getting Started](docs/getting-started.dox) - installation and build options
- [User Guide](docs/user-guide.dox) - API usage for C++, C, and Fortran
- [Linking Guide](docs/linking.dox) - compile and link instructions
- [Backend Reference](docs/backends.dox) - backend-specific configuration
- [Limitations](docs/limitations.dox) - current constraints

Build local HTML documentation (optional):

```bash
doxygen docs/Doxyfile
```

Then open `docs/html/index.html` in your browser.

## License

MIT License. See [LICENSE](./LICENSE) for details.

## Citation

If you use SHAFFT in your research, please cite the underlying method:

```bibtex
@article{dalcin2019fast,
  title={Fast parallel multidimensional FFT using advanced MPI},
  author={Dalcin, Lisandro and Mortensen, Mikael and Keyes, David E},
  journal={Journal of Parallel and Distributed Computing},
  volume={128},
  pages={137--150},
  year={2019},
  doi={10.1016/j.jpdc.2019.02.006}
}
```
