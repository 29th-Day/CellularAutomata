#pragma once

#include "common.h"

// function decoration if CUDA is used to be usable in kernel
#ifdef CUDA
#include <cuda_runtime.h>
#define cuda __device__
#else
#define cuda
#endif

namespace Activations
{
    cuda float sigmoid(float x);

    cuda float life(float x);

    cuda float clip(float x);

    cuda float sin(float x);

    cuda float cos(float x);

    cuda float tan(float x);
}
