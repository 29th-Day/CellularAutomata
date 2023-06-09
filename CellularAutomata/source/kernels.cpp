#include "kernels.h"
#include "common.h"
#include "rng.h"

#include <stdlib.h>

template <typename T>
void Kernels::ones(T *array, const int s)
{
    for (int i = 0; i < s * s; i++)
    {
        array[i] = static_cast<T>(1);
    }
}

template void Kernels::ones<float>(float *, const int);

template <typename T>
void Kernels::life(T *array, const int s)
{
    int center = (s * s) / 2;
    for (int i = 0; i < s * s; i++)
    {
        if (i == center)
            array[i] = static_cast<T>(16);
        else
            array[i] = static_cast<T>(1);
    }
}

template void Kernels::life<float>(float *, const int);

// void Kernels::rand(Kernel *kernel)
// {
//     for (int i = 0; i < kernel_size; i++)
//     {
//         kernel->kernel[i] = RNG::decimal(-1, 1);
//     }
// }

// void Kernels::normal(Kernel *kernel)
// {
//     for (int i = 0; i < kernel_size; i++)
//     {
//         kernel->kernel[i] = RNG::decimal(0, 1);
//     }
// }
