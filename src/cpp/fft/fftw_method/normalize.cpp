#include "normalize.h"
#include <fftw3.h>
#include <cstddef>

int normalizeComplexFloat(float norm_factor,
                          std::size_t tensor_size,
                          void* data)
{
    if (!data || tensor_size == 0) return 0;

    auto* arr = reinterpret_cast<fftwf_complex*>(data);

    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(tensor_size); ++i) {
        arr[i][0] *= norm_factor;
        arr[i][1] *= norm_factor;
    }
    return 0;
}

int normalizeComplexDouble(double norm_factor,
                           std::size_t tensor_size,
                           void* data)
{
    if (!data || tensor_size == 0) return 0;

    auto* arr = reinterpret_cast<fftw_complex*>(data);

    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(tensor_size); ++i) {
        arr[i][0] *= norm_factor;
        arr[i][1] *= norm_factor;
    }
    return 0;
}
