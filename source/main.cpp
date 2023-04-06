#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "activations.h"
#include "kernels.h"
#include "states.h"
#include "display.h"

#include "engine.h"

#define ASSERTION_ERROR -2

#define assert(x, msg) {if (!(x)) {fprintf(stderr, "Assertion error: %s", msg); exit(ASSERTION_ERROR);}}



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


void parseArgs(int argc, char** argv, int *height, int *width, int *scale, int *fps)
{
    for(int i = 1; i < argc; i+=2)
    {
        if(strcmp(argv[i], "-h") == 0)
            *height = atoi(argv[i+1]);
        else if (strcmp(argv[i], "-w") == 0)
            *width = atoi(argv[i+1]);
        else if (strcmp(argv[i], "-s") == 0)
            *scale = atoi(argv[i+1]);
        else if (strcmp(argv[i], "-fps") == 0)
            *fps = atoi(argv[i+1]);
        else if (strcmp(argv[i], "-seed") == 0)
            srand((unsigned int)*argv[i+1]);
    }
}

int main(int argc, char** argv)
{
    // VARIABLES

    int HEIGHT      = 100;
    int WIDTH       = 100;
    int SCALE       = 1;
    int FPS         = 10;

    int KERNEL_SIZE = 3;

    // SETUP

    parseArgs(argc, argv, &HEIGHT, &WIDTH, &SCALE, &FPS);

    assert(HEIGHT >= 1, "HEIGHT must be greater than 0");
    assert(WIDTH >= 1, "WIDTH must be greater than 0");
    assert(SCALE >= 1, "SCALE must be greater than 0");
    assert(FPS >= 1, "FPS must be greater than 0");

    Display display = Display(HEIGHT, WIDTH, SCALE, FPS);

    LifeEngine engine = LifeEngine(HEIGHT, WIDTH, State::randb, Kernel::rand, Activation::identity);


    // MAIN PART

    while(display.run())
    {
        if (display.nextFrame())
        {
            display.draw(engine.state);
            engine.epoch();
        }

    }

    return 0;
}
