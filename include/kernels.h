#pragma once

typedef float (*kernel_func)(float**);

namespace Kernel
{
    int life(float** kernel);

    int half(float **kernel);
}