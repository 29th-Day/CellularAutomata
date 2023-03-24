#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "activations.h"
#include "kernels.h"
#include "display.h"

#define INVALID_ARGS -2

#define SWAP_PTR(a, b) {float **temp = b; b = a; a = temp;}


void print2D(float **array, int height, int width)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            printf("%.1f ", array[y][x]);
        }
        printf("\n");
    }
}

float **init2D(int height, int width, float value = NULL)
{
    float **array = new float*[height];
    for (int y = 0; y < height; y++)
    {
        array[y] = new float[width]();

        if (value != NULL)
        {
            for (int x = 0; x < width; x++)
                array[y][x] = value;
        }
    }
    return array;
}

void free2D(float **array, int height)
{
    for (int y = 0; y < height; y++)
        delete[] array[y];
    delete[] array;
}

void convolution(float **src, float **dest, int height, int width, float **kernel, int size, activation_func f)
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

                    sum += src[array_y][array_x] * kernel[y][x];
                }

                // printf("\n");
            }
            
            dest[row][col] = f(sum);
            sum = 0.0f;
        }
    }
}

int main(int argc, char** argv)
{

    // Display display = Display(100, 100);

    // while (!display.close)
    // {
    //     display.draw();
    // }
    
    // exit(0);


    // VARIABLES

    int HEIGHT = 10;
    int WIDTH = 10;

    int KERNEL_SIZE = 3;

    int ITERATIONS = 5;

    // SETUP

    if (argc == 3)
    {
        HEIGHT = atoi(argv[1]);
        WIDTH = atoi(argv[2]);
    }

    if (HEIGHT < 1 || WIDTH < 1)
    {
        fprintf(stderr, "HEIGHT and WIDTH must be greater than zero");
        return INVALID_ARGS;
    }

    printf("Cellular Automata: %ux%u\n", HEIGHT, WIDTH);


    float **current = init2D(HEIGHT, WIDTH);
    float **next = init2D(HEIGHT, WIDTH);


    // float **kernel = init2D(KERNEL_SIZE, KERNEL_SIZE, 1.0f);
    float **kernel = Kernel::life(&KERNEL_SIZE);

    current[4][3] = 1.0f;
    current[4][4] = 1.0f;
    current[4][5] = 1.0f;
    current[5][4] = 1.0f;

    // MAIN PART

    for (int i = 0; i < ITERATIONS; i++)
    {
        print2D(current, HEIGHT, WIDTH);
        printf("\n");

        convolution(current, next, HEIGHT, WIDTH, kernel, KERNEL_SIZE, Activation::life);

        SWAP_PTR(current, next);
    }

    // CLEAN UP

    free2D(kernel, KERNEL_SIZE);
    free2D(current, HEIGHT);
    free2D(next, HEIGHT);

    return 0;
}