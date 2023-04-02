#include "engine.h"

#include "kernels.h"
#include "activations.h"
#include "states.h"

#define SWAP(a, b) {float *temp = b; b = a; a = temp;}


void convolution(float src[], float dest[], int height, int width, float kernel[], int size, activation_func f)
{
    int relative_y = 0;
    int relative_x = 0;

    int array_y = 0;
    int array_x = 0;

    float sum = 0;

    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            
            
            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    relative_y = y - (size / 2);
                    relative_x = x - (size / 2);

                    array_y = row + relative_y;
                    array_x = col + relative_x;

                    // printf("(%2i, %2i) ", array_y, array_x);

                    if (array_y < 0 || array_y >= height)
                        continue; // add nothing
                    
                    if (array_x < 0 || array_x >= width)
                        continue; // add nothing

                    sum += src[array_y * width + array_x] * kernel[y * size + x];
                }

                // printf("\n");
            }
            
            dest[row * width + col] = f(sum);
            sum = 0.0f;
        }
    }
}


LifeEngine::LifeEngine(int height, int width, state_func sf, kernel_func kf, activation_func af)
{
    HEIGHT = height;
    WIDTH = width;
    iteration = 0;

    state = new float[HEIGHT * WIDTH]();
    next = new float[HEIGHT * WIDTH]();

    sf(state, HEIGHT, WIDTH);

    kernel = nullptr;
    KERNEL_SIZE = kf(&kernel);

    fn = af;
}

LifeEngine::~LifeEngine()
{
    delete[] state;
    delete[] next;
    delete[] kernel;
}

void LifeEngine::epoch()
{
    convolution(state, next, HEIGHT, WIDTH,
        kernel, KERNEL_SIZE, fn);

    SWAP(state, next);

    iteration++;
}