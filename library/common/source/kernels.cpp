#include "kernels.h"
#include "common.h"
#include "rng.h"

#include <stdlib.h>

#define kernel_size kernel->size * kernel->size

void Kernels::full(Kernel *kernel)
{
    for (int i = 0; i < kernel_size; i++)
    {
        kernel->kernel[i] = 1.0f;
    }
}

void Kernels::life(Kernel *kernel)
{
    int center = kernel_size / 2;
    for (int i = 0; i < kernel_size; i++)
    {
        if (i == center)
            kernel->kernel[i] = 0.5f;
        else
            kernel->kernel[i] = 1.0f;
    }
}

void Kernels::half(Kernel *kernel)
{
    for (int i = 0; i < kernel_size; i++)
    {
        kernel->kernel[i] = 0.5f;
    }
}

void Kernels::rand(Kernel *kernel)
{
    for (int i = 0; i < kernel_size; i++)
    {
        kernel->kernel[i] = RNG::decimal(-1, 1);
    }
}

void Kernels::randp(Kernel *kernel)
{
    for (int i = 0; i < kernel_size; i++)
    {
        kernel->kernel[i] = RNG::decimal(0, 1);
    }
}
