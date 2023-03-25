#include "kernels.h"

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