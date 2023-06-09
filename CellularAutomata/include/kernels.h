#pragma once

#include "common.h"

namespace Kernels
{
    template <typename T>
    void ones(T *array, const int s);

    template <typename T>
    void life(T *array, const int s);

    // void life(double *array, const int s);

    // void half(float *array, const int s);
    // void half(double *array, const int s);

    // void rand(Kernel *kernel);

    // void normal(Kernel *kernel);
}
