#pragma once

#include "common.h"

#ifdef cudaEnabled
#include <cuda_runtime.h>
#define cudaFn __device__ __host__
#else
#define cudaFn
#endif

namespace Activations
{
    template <typename T>
    cudaFn T _normal(T x);

    template <typename T>
    cudaFn T _life(T x);

    template <typename T>
    cudaFn T _sigmoid(T x);

    template <typename T>
    cudaFn T _tanh(T x);

    template <typename T>
    cudaFn T _sin(T x);

    template <typename T>
    cudaFn T _cos(T x);

    template <typename T>
    cudaFn T _tan(T x);
}
