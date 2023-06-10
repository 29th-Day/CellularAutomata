#include "CellularAutomata.h"

#include "backend_base.h"
#include "common.h"
#include "rng.h"
#include <algorithm>

#include <cstdio>

#ifdef cudaEnabled
#include "backend_cuda.h"
#endif

unsigned int CellularAutomata::InitRandom(unsigned int seed)
{
    return RNG::seed(0);
}

template <typename T>
void CellularAutomata::InitState(State<T> *state, const int h, const int w, stateFunc<T> fn, const Device device)
{
#ifdef cudaEnabled
    size_t size = h * w * sizeof(T);

    if (device == Device::CUDA)
    {
        // allocate GPU memory

        allocateCUDA(reinterpret_cast<void **>(&state->curr), size);
        allocateCUDA(reinterpret_cast<void **>(&state->next), size);

        if (fn != nullptr)
        {
            T *temp = new T[h * w]();

            fn(temp, h, w);
            copyCUDA(temp, Device::CPU, state->curr, Device::CUDA, size);

            delete[] temp;
        }
    }
    else
    {
        // allocate (pinned) CPU memory (for faster transfer)
        // just always use pinned memory when CUDA is enabled (maybe use condition?)

        allocateHost(reinterpret_cast<void **>(&state->curr), size);
        allocateHost(reinterpret_cast<void **>(&state->next), size);

        if (fn != nullptr)
            fn(state->curr, h, w);
    }
#else
    // allocate CPU memory

    state->curr = new T[h * w]();
    state->next = new T[h * w]();

    if (fn != nullptr)
        fn(state->curr, h, w);
#endif

    state->height = h;
    state->width = w;
    state->device = device;
}

template void CellularAutomata::InitState(State<float> *, int, int, stateFunc<float>, Device);

template <typename T>
void CellularAutomata::DestroyState(State<T> *state)
{
    if (state->device == Device::CUDA)
    {
#ifdef cudaEnabled

        // free GPU memory

        freeCUDA(state->curr);
        freeCUDA(state->next);
#endif
    }
    else
    {
#ifdef cudaEnabled
        // free (pinned) CPU memory

        freeHost(state->curr);
        freeHost(state->next);
#else
        // free CPU memory

        delete[] state->curr;
        delete[] state->next;
#endif
    }

    state->curr = nullptr;
    state->next = nullptr;
    state->height = 0;
    state->width = 0;
}

template void CellularAutomata::DestroyState<float>(State<float> *);

template <typename T>
void CellularAutomata::InitKernel(Kernel<T> *kernel, int s, kernelFunc<T> fn, Device device)
{
    if (device == Device::CUDA)
    {
        // allocate GPU memory

#ifdef cudaEnabled

        size_t size = s * s * sizeof(T);

        allocateCUDA(reinterpret_cast<void **>(&kernel->kernel), size);

        if (fn != nullptr)
        {
            T *temp = new T[s * s]();

            fn(temp, s);
            copyCUDA(temp, Device::CPU, kernel->kernel, Device::CUDA, size);

            delete[] temp;
        }

#endif
    }
    else
    {
        // allocate CPU memory
        // (no pinned needed, because it isn't transferred often)

        kernel->kernel = new T[s * s]();

        if (fn != nullptr)
            fn(kernel->kernel, s);
    }

    kernel->size = s;
}

template void CellularAutomata::InitKernel(Kernel<float> *, int, kernelFunc<float>, Device);

template <typename T>
void CellularAutomata::DestroyKernel(Kernel<T> *kernel)
{
    if (kernel->device == Device::CUDA)
    {
        // free GPU memory

#ifdef cudaEnabled

        freeCUDA(kernel->kernel);

#endif
    }
    else
    {
        // free CPU memory

        delete[] kernel->kernel;
    }

    kernel->kernel = nullptr;
    kernel->size = 0;
}

template void CellularAutomata::DestroyKernel(Kernel<float> *);

template <typename T>
void CellularAutomata::CopyTo(State<T> *src, State<T> *dst)
{

#ifdef cudaEnabled
    // copy with CUDA arrays

    size_t size = src->height * src->width * sizeof(T);
    copyCUDA(src->curr, src->device, dst->curr, dst->device, size);
#else
    // copy with standard arrays

    std::copy(src->curr, src->curr + src->height * src->width, dst->curr);
#endif
}

template void CellularAutomata::CopyTo(State<float> *, State<float> *);

template <typename T>
void CellularAutomata::CopyTo(Kernel<T> *src, Kernel<T> *dst)
{
#ifdef cudaEnabled
    // copy with CUDA arrays

    size_t size = src->size * src->size * sizeof(T);
    copyCUDA(src->kernel, src->device, dst->kernel, dst->device, size);
#else
    // copy with standard arrays

    std::copy(src->kernel, src->kernel + src->size * src->size, dst->kernel);
#endif
}

template void CellularAutomata::CopyTo(Kernel<float> *, Kernel<float> *);

#define swap(a, b) \
    auto tmp = a;  \
    a = b;         \
    b = tmp;

template <typename T>
void CellularAutomata::Epoch(State<T> *state, Kernel<T> *kernel, Activations::OpCode activation, bool recursive)
{
    if (state->device == Device::CUDA)
    {
        // GPU convolution

#ifdef cudaEnabled

        epochCUDA(
            state->curr, kernel->kernel, state->next, activation,
            state->height, state->width, kernel->size);

#endif
    }
    else
    {
        // CPU convolution

        epochCPU(
            state->curr, kernel->kernel, state->next, activation,
            state->height, state->width, kernel->size, recursive);
    }

    swap(state->curr, state->next)
}

template void CellularAutomata::Epoch(State<float> *, Kernel<float> *, Activations::OpCode, bool);
