#include "kernels.h"
#include "common.h"

#include <stdlib.h>

inline float random()
{
    return ((rand() % 2) ? 1 : -1) * (float)rand() / (float)(RAND_MAX);
}

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
        random(), random(), random(),
        random(), random(), random(),
        random(), random(), random()};
    kernel->size = 3;
}