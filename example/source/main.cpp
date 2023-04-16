#include <stdio.h>
#include <stdlib.h>

#include "engine.h"
#include "display.h"

#define assert(x, msg)                                   \
    {                                                    \
        if (!(x))                                        \
        {                                                \
            fprintf(stderr, "Assertion error: %s", msg); \
            exit(-2);                                    \
        }                                                \
    }

#define EQUAL_S(a, b) strcmp(a, b) == 0

struct Arguments
{
    int height;
    int width;
    int scale;
    int fps;
    int seed;
};

void parseArgs(int argc, char **argv, Arguments *args)
{
    for (int i = 1; i < argc; i += 2)
    {
        if (EQUAL_S(argv[i], "-h"))
            args->height = atoi(argv[i + 1]);
        else if (EQUAL_S(argv[i], "-w"))
            args->width = atoi(argv[i + 1]);
        else if (EQUAL_S(argv[i], "-s"))
            args->scale = atoi(argv[i + 1]);
        else if (EQUAL_S(argv[i], "-fps"))
            args->fps = atoi(argv[i + 1]);
        else if (EQUAL_S(argv[i], "-seed"))
            args->seed = atoi(argv[i + 1]);
    }

    assert(args->height > 0, "HEIGHT must be greater than 0");
    assert(args->width > 0, "WIDTH must be greater than 0");
    assert(args->scale > 0, "SCALE must be greater than 0");
    assert(args->fps > 0, "FPS must be greater than 0");
}

int main(int argc, char **argv)
{
    // SETUP

    Arguments args{0};
    parseArgs(argc, argv, &args);

    args.seed = Engine::InitRandom(args.seed);
    printf("seed: %u\n", args.seed);

    State state;
    Kernel kernel;

    Engine::InitState(&state, args.height, args.width, States::randb);
    Engine::InitKernel(&kernel, Kernels::life);

    // MAIN PART

    Display display = Display(args.height, args.width, args.scale, args.fps);
    while (display.run())
    {
        if (display.nextFrame())
        {
            display.draw(state.current);
            Engine::Epoch(&state, &kernel, Activations::life);
        }
    }

    Engine::DestroyState(&state);
    Engine::DestroyKernel(&kernel);

    return 0;
}
