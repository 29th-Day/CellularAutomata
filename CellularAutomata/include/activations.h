#pragma once

#include "common.h"

// #ifdef cudaEnabled
// #include <cuda_runtime.h>
// // #define cudaFn #ifdef __CUDACC__ __device__ __host #endif
// #endif

#include <cuda_runtime.h>

namespace Activations
{
    template <typename T>
    __device__ __host__ T _normal(T x);

    template <typename T>
    __device__ __host__ T _life(T x);

    // template <typename T>
    // device host T sigmoid(T x);

    // template <typename T>
    // device host T sin(T x);

    // template <typename T>
    // device host T cos(T x);

    // template <typename T>
    // device host T tan(T x);

}
