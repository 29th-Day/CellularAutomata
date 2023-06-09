#include "activations.h"

#include <cmath>

#include <cuda_runtime.h>

#define EULER_NUMBER_F 2.71828182846f

template <typename T>
__device__ __host__ T Activations::_normal(T x)
{
    if (x < 0)
        return static_cast<T>(0);
    else if (x > 1)
        return static_cast<T>(1);
    else
        return x;
}

template __device__ __host__ float Activations::_normal(float);

template <typename T>
__device__ __host__ T Activations::_life(T x)
{
    unsigned char u = static_cast<unsigned char>(x);

    unsigned char neighbours = u & 0xF; // low  nibble
    bool alive = u & 0xF0;              // high nibble

    switch (neighbours)
    {
    case 2:
        // staying alive
        return static_cast<T>((alive) ? 1 : 0);
    case 3:
        // birth
        return static_cast<T>(1);
    default:
        // under- / overpopulation
        return static_cast<T>(0);
    }
}

template __device__ __host__ float Activations::_life(float);

// template <typename T>
// device host T Activations::sigmoid(T x)
// {
//     return (1 / (1 + std::pow(EULER_NUMBER_F, -x)));
// }

// template <typename T>
// device host T Activations::sin(T x)
// {
//     return std::sin(x);
// }

// template <typename T>
// device host T Activations::cos(T x)
// {
//     return std::cos(x);
// }

// template <typename T>
// device host T Activations::tan(T x)
// {
//     return std::tan(x);
// }
