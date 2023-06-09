#pragma once

#include "common.h"

void allocateCUDA(void **array, size_t bytes);

void allocateHost(void **array, size_t bytes);

void freeCUDA(void *array);

void freeHost(void *array);

void copyCUDA(void *src, Device from, void *dst, Device to, size_t bytes);

template <typename T>
void epochCUDA(
    T *input, T *kernel, T *output, const int op,
    const int h, const int w, const int s);
