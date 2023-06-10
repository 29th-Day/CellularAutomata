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

    // template <typename T>
    // device host T sigmoid(T x);

    // template <typename T>
    // device host T sin(T x);

    // template <typename T>
    // device host T cos(T x);

    // template <typename T>
    // device host T tan(T x);

}
