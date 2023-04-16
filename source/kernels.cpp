#include "kernels.h"
#include "common.h"
#include "rng.h"

#include <stdlib.h>

void Kernels::life(Kernel *kernel)
{
    kernel->kernel = new float[3 * 3]{
        1.0f, 1.0f, 1.0f,
        1.0f, 0.5f, 1.0f,
        1.0f, 1.0f, 1.0f};
    kernel->size = 3;
}

void Kernels::half(Kernel *kernel)
{
    kernel->kernel = new float[3 * 3]{
        0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f};
    kernel->size = 3;
}

void Kernels::rand(Kernel *kernel)
{
    kernel->kernel = new float[3 * 3]{
        RNG::decimal(), RNG::decimal(), RNG::decimal(),
        RNG::decimal(), RNG::decimal(), RNG::decimal(),
        RNG::decimal(), RNG::decimal(), RNG::decimal()};
    kernel->size = 3;
}