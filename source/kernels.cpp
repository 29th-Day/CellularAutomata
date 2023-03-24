#include "kernels.h"

float **Kernel::life(int* KERNEL_SIZE)
{
    float **array = new float*[3];
    for (int y = 0; y < 3; y++)
    {
        array[y] = new float[3] {1.0f, 1.0f, 1.0f};
    }
    array[1][1] = 0.5f;

    *KERNEL_SIZE = 3;

    return array;
}