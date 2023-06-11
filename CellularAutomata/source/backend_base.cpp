#include "backend_base.h"

#include "activations.h"
#include "common.h"

using namespace Activations;

template <typename T>
activationFunc<T> activFunc[] = {_normal, _life, _sigmoid, _tanh, _sin, _cos, _tan};

template <typename T>
void epochCPU(
    const T *input, const T *kernel, T *output, int op,
    const int h, const int w, const int s, const bool r)
{
    int kernel_radius = s / 2;

#pragma omp parallel for
    // iterate over state array
    for (int row = 0; row < h; row++)
    {
        // definitons inside first loop for parallelization
        int array_y = 0;
        int array_x = 0;
        float sum = 0.0f;

        for (int col = 0; col < w; col++)
        {

            // iterate over kernel
            for (int y = 0; y < s; y++)
            {
                for (int x = 0; x < s; x++)
                {
                    // calculate state array positions
                    array_y = row + (y - kernel_radius);
                    array_x = col + (x - kernel_radius);

                    if (r)
                    {
                        // overflow y
                        if (array_y < 0)
                            array_y += h;
                        else if (array_y >= h)
                            array_y -= h;
                        // overflow x
                        if (array_x < 0)
                            array_x += w;
                        else if (array_x >= w)
                            array_x -= w;
                    }
                    else
                    {
                        // overflow on y or x
                        if (array_y < 0 || array_y >= h || array_x < 0 || array_x >= w)
                            continue; // add nothing
                    }

                    // State x Kernel
                    sum += input[array_y * w + array_x] * kernel[y * s + x];
                }
            }

            // Set new value
            output[row * w + col] = activFunc<T>[op](sum);
            sum = 0.0f;
        }
    }
}
