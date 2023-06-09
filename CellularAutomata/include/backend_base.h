#pragma once

#include "common.h"

template <typename T>
void epochCPU(
    const T *input, const T *kernel, T *output, int op,
    const int h, const int w, const int s, const bool r);

template void epochCPU<float>(
    const float *, const float *, float *, int op,
    const int, const int, const int, const bool);
