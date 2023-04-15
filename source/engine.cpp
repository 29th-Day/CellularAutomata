#include "engine.h"
#include "common.h"

#include <stdlib.h>

#define SWAP(a, b)     \
    {                  \
        auto temp = b; \
        b = a;         \
        a = temp;      \
    }

// void print2D(float *array, int height, int width)
// {
//     for (int y = 0; y < height; y++)
//     {
//         for (int x = 0; x < width; x++)
//         {
//             printf("%4.1f ", array[y * width + x]);
//         }
//         printf("\n");
//     }
//     printf("\n");
// }

// void convolution(float src[], float dest[], int height, int width, float kernel[], int size, activation_func f)
// {
//     int relative_y = 0;
//     int relative_x = 0;
//
//     int array_y = 0;
//     int array_x = 0;
//
//     float sum = 0;
//
//     for (int row = 0; row < height; row++)
//     {
//         for (int col = 0; col < width; col++)
//         {
//
//             for (int y = 0; y < size; y++)
//             {
//                 for (int x = 0; x < size; x++)
//                 {
//                     relative_y = y - (size / 2);
//                     relative_x = x - (size / 2);
//
//                     array_y = row + relative_y;
//                     array_x = col + relative_x;
//
//                     // printf("(%2i, %2i) ", array_y, array_x);
//
//                     if (array_y < 0 || array_y >= height)
//                         continue; // add nothing
//
//                     if (array_x < 0 || array_x >= width)
//                         continue; // add nothing
//
//                     sum += src[array_y * width + array_x] * kernel[y * size + x];
//                 }
//
//                 // printf("\n");
//             }
//
//             dest[row * width + col] = f(sum);
//             sum = 0.0f;
//         }
//     }
// }

void Engine::InitRandom(unsigned int seed)
{
    srand(seed);
}

void Engine::InitState(State *state, int height, int width, state_func f)
{
    // state = new float[HEIGHT * WIDTH]();
    state->height = height;
    state->width = width;
    state->current = new float[height * width]();
    state->next = new float[height * width]();
    f(state);
}

void Engine::DestroyState(State *state)
{
    delete[] state->current;
    state->current = nullptr;
    delete[] state->next;
    state->current = nullptr;
}

void Engine::InitKernel(Kernel *kernel, kernel_func f)
{
    f(kernel);
}

void Engine::DestroyKernel(Kernel *kernel)
{
    delete[] kernel->kernel;
    kernel->kernel = nullptr;
}

void Engine::Epoch(State *state, Kernel *kernel, activation_func f)
{
    // convolution
    int relative_y = 0;
    int relative_x = 0;

    int array_y = 0;
    int array_x = 0;

    float sum = 0.0f;

    // iterate over state array
    for (int row = 0; row < state->height; row++)
    {
        for (int col = 0; col < state->width; col++)
        {

            // iterate over kernel
            for (int y = 0; y < kernel->size; y++)
            {
                for (int x = 0; x < kernel->size; x++)
                {
                    // calculate relative position to middle
                    relative_y = y - (kernel->size / 2);
                    relative_x = x - (kernel->size / 2);

                    // calculate state array positions
                    array_y = row + relative_y;
                    array_x = col + relative_x;

                    // recursion / nothing
                    if (array_y < 0 || array_y >= state->height)
                        continue; // add nothing

                    if (array_x < 0 || array_x >= state->width)
                        continue; // add nothing

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