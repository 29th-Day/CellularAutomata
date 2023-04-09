#include "kernels.h"

#include <stdlib.h>

inline float random()
{
    return ((rand() % 2) ? 1 : -1) * (float)rand()/(float)(RAND_MAX);
}

int Kernel::life(float** kernel)
{
    *kernel = new float[3 * 3] {
        1.0f, 1.0f, 1.0f,
        1.0f, 0.5f, 1.0f,
        1.0f, 1.0f, 1.0f
    };
    return 3;
}

int Kernel::half(float** kernel)
{
    *kernel = new float[3 * 3] {
        0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f
    };
    return 3;
}

int Kernel::rand(float** kernel)
{
    *kernel = new float[3 * 3] {
        random(), random(), random(),
        random(), random(), random(),
        random(), random(), random()
    };
    return 3;
}