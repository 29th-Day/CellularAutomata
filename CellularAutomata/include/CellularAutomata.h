#pragma once

#include "activations.h"
#include "common.h"
#include "kernels.h"
#include "rng.h"
#include "states.h"

namespace CellularAutomata
{
    unsigned int InitRandom(unsigned int seed);

    template <typename T>
    void InitState(State<T> *state, int h, int w, stateFunc<T> fn, Device device);

    template <typename T>
    void DestroyState(State<T> *state);

    template <typename T>
    void InitKernel(Kernel<T> *kernel, int s, kernelFunc<T> fn, Device device);

    template <typename T>
    void DestroyKernel(Kernel<T> *kernel);

    template <typename T>
    void CopyTo(State<T> *src, State<T> *dst);

    template <typename T>
    void CopyTo(Kernel<T> *src, Kernel<T> *dst);

    template <typename T>
    void Epoch(State<T> *state, Kernel<T> *kernel, Activations::OpCode activation, bool recursive);
}
