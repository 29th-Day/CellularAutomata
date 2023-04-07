#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "activations.h"
#include "kernels.h"
#include "states.h"
#include "display.h"

#include "engine.h"

#define assert(x, msg) {if (!(x)) {fprintf(stderr, "Assertion error: %s", msg); exit(-2);}}

#define EQUAL_S(a, b) strcmp(a, b) == 0

struct Arguments {
    int height;
    int width;
    int scale;
    int fps;
    int seed;
};

void parseArgs(int argc, char** argv, Arguments *args)
{
    args->seed = (int)time(NULL);

    for(int i = 1; i < argc; i+=2)
    {
        if(EQUAL_S(argv[i], "-h"))
            args->height = atoi(argv[i+1]);
        else if (EQUAL_S(argv[i], "-w"))
            args->width = atoi(argv[i+1]);
        else if (EQUAL_S(argv[i], "-s"))
            args->scale = atoi(argv[i+1]);
        else if (EQUAL_S(argv[i], "-fps"))
            args->fps = atoi(argv[i+1]);
        else if (EQUAL_S(argv[i], "-seed"))
            args->seed = atoi(argv[i+1]);
    }

    assert(args->height > 0, "HEIGHT must be greater than 0");
    assert(args->width > 0, "WIDTH must be greater than 0");
    assert(args->scale > 0, "SCALE must be greater than 0");
    assert(args->fps > 0, "FPS must be greater than 0");
}

int main(int argc, char** argv)
{
    // SETUP
    Arguments args;
    parseArgs(argc, argv, &args);

    srand(args.seed);
    printf("seed: %i\n", args.seed);

    Display display = Display(args.height, args.width, args.scale, args.fps);

    LifeEngine engine = LifeEngine(args.height, args.width,
        State::randb, Kernel::life, Activation::life);


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
