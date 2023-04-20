#include "engine.h"
#include "common.h"
#include "rng.h"

#include <stdlib.h>
#include <stdio.h>

#ifdef OPENMP
#include <omp.h>
#endif

#define SWAP(a, b)     \
    {                  \
        auto temp = b; \
        b = a;         \
        a = temp;      \
    }

unsigned int Engine::InitRandom(unsigned int seed)
{
    return RNG::seed(seed);
}

void Engine::InitState(State *state, int height, int width, state_func f)
{
    state->height = height;
    state->width = width;
    state->current = new float[height * width]();
    state->next = new float[height * width]();
    if (f != NULL)
        f(state);
}

void Engine::DestroyState(State *state)
{
    delete[] state->current;
    state->current = nullptr;
    delete[] state->next;
    state->next = nullptr;
    state->height = 0;
    state->width = 0;
}

void Engine::InitKernel(Kernel *kernel, kernel_func f)
{
    f(kernel);
}

void Engine::DestroyKernel(Kernel *kernel)
{
    delete[] kernel->kernel;
    kernel->kernel = nullptr;
    kernel->size = 0;
}

void Engine::Epoch(State *state, Kernel *kernel, activation_func f, bool recursive)
{

// iterate over state array
#ifdef OPENMP
#pragma omp parallel for
#endif
    for (int row = 0; row < state->height; row++)
    {
        // definitons inside first loop for parallelization
        int kernel_center = kernel->size / 2;
        int array_y = 0;
        int array_x = 0;
        float sum = 0.0f;

        for (int col = 0; col < state->width; col++)
        {

            // iterate over kernel
            for (int y = 0; y < kernel->size; y++)
            {
                for (int x = 0; x < kernel->size; x++)
                {
                    // calculate state array positions
                    array_y = row + (y - kernel_center);
                    array_x = col + (x - kernel_center);

                    if (recursive)
                    {
                        // overflow y
                        if (array_y < 0)
                            array_y = state->height + array_y;
                        else if (array_y >= state->height)
                            array_y = array_y - state->height;
                        // overflow x
                        if (array_x < 0)
                            array_x = state->width + array_x;
                        else if (array_x >= state->width)
                            array_x = array_x - state->width;
                    }
                    else
                    {
                        // overflow y
                        if (array_y < 0 || array_y >= state->height)
                            continue; // add nothing
                        // overflow x
                        if (array_x < 0 || array_x >= state->width)
                            continue; // add nothing
                    }

                    // State x Kernel
                    sum += state->current[array_y * state->width + array_x] * kernel->kernel[y * kernel->size + x];
                }
            }

            // Set new value
            state->next[row * state->width + col] = f(sum);
            sum = 0.0f;
        }
    }

    // swap
    SWAP(state->current, state->next);
}
