#include "epoch.hpp"
#include "cpu.hpp"
#include "exceptions.hpp"

#ifdef __CUDACC__
#include "../cuda/cuda.cuh"
#endif

namespace CellularAutomata
{
    template <typename T, typename Activation>
    void Epoch(State<T>* state, Kernel<T>* kernel, Activation activation, bool recursive)
    {
        if (state->device != kernel->device)
            throw exceptions::DevicesUnequal("Devices of state and kernel are unequal");

        if (state->device == Device::CUDA)
        {
        #ifdef __CUDACC__
            CellularAutomata::cuda::epoch(
                state->curr, kernel->kernel, state->next, activation,
                state->height, state->width, kernel->size, recursive
            );
        #endif
        }
        else
        {
            CellularAutomata::cpu::epoch(
                state->curr, kernel->kernel, state->next, activation,
                state->height, state->width, kernel->size, recursive
            );
        }

        state->swap();
        }
    }
