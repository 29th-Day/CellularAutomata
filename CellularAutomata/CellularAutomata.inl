#include "CellularAutomata.hpp"
#include "cpu.hpp"

#ifdef __CUDACC__
#include "cuda.cuh"
#endif

namespace CellularAutomata
{
    template <typename T, typename Activation>
    void Epoch(State<T>* state, Kernel<T>* kernel, Activation activation, bool recursive)
    {
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
