#pragma once

struct State
{
    float *current = nullptr;
    float *next = nullptr;
    int height;
    int width;
};

struct Kernel
{
    float *kernel = nullptr;
    int size;
};

typedef void (*kernel_func)(Kernel *);

typedef void (*state_func)(State *);

typedef float (*activation_func)(float);

#define SWAP(a, b)     \
    {                  \
        auto temp = b; \
        b = a;         \
        a = temp;      \
    }
