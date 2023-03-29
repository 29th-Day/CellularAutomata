#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "activations.h"
#include "kernels.h"
#include "states.h"
#include "display.h"

#define INVALID_ARGS -2

#define SWAP_PTR(a, b) {float *temp = b; b = a; a = temp;}


void print2D(float *array, int height, int width)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            printf("%.1f ", array[y * width + x]);
        }
        printf("\n");
    }
    printf("\n");
}

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


int main(int argc, char** argv)
{
    // VARIABLES

    int HEIGHT      = 100;
    int WIDTH       = 100;
    int SCALE       = 1;
    int FPS         = 10;
    // int ITERATIONS  = 5;

    int KERNEL_SIZE = 3;

    // SETUP

    for(int i = 1; i < argc; i+=2)
    {
        if(strcmp(argv[i], "-h") == 0)
            HEIGHT = atoi(argv[i+1]);
        else if (strcmp(argv[i], "-w") == 0)
            WIDTH = atoi(argv[i+1]);
        else if (strcmp(argv[i], "-s") == 0)
            SCALE = atoi(argv[i+1]);
        else if (strcmp(argv[i], "-fps") == 0)
            FPS = atoi(argv[i+1]);
        else if (strcmp(argv[i], "-seed") == 0)
            srand((unsigned int)*argv[i+1]);
            
    }

    if (HEIGHT < 1 || WIDTH < 1)
    {
        fprintf(stderr, "HEIGHT and WIDTH must be greater than zero");
        return INVALID_ARGS;
    }

    // state
    float *current = new float[HEIGHT * WIDTH]();
    float *next = new float[HEIGHT * WIDTH]();

    State::randb(current, HEIGHT, WIDTH);

    // current[49 * WIDTH + 48] = 1.0f;
    // current[49 * WIDTH + 49] = 1.0f;
    // current[49 * WIDTH + 50] = 1.0f;
    // current[50 * WIDTH + 49] = 1.0f;


    float *kernel = nullptr;
    
    // KERNEL_SIZE = Kernel::half(&kernel);
    KERNEL_SIZE = Kernel::life(&kernel);

    Display display = Display(HEIGHT, WIDTH, SCALE, FPS);

    // MAIN PART

    display.draw(current);
    while(display.run())
    {
        convolution(current, next, HEIGHT, WIDTH,
            kernel, KERNEL_SIZE, Activation::life);

        SWAP_PTR(current, next);

        display.draw(current);
    }

    // CLEAN UP

    delete[] current;
    delete[] next;
    delete[] kernel;

    return 0;
}
