#pragma once

typedef int (*kernel_func)(float**);

namespace Kernel
{
    int life(float** kernel);

    int half(float **kernel);

    int rand(float** kernel);
}