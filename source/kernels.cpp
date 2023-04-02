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

int Kernel::test(float** kernel)
{
    *kernel = new float[3 * 3] {
        0.1f, 0.1f, 0.1f,
        0.1f, -1.0f, 0.1f,
        0.1f, 0.1f, 0.1f
    };
    return 3;
}