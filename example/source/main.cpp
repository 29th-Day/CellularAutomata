/**
 * @file main.cpp
 * @author 29th-Day (https://github.com/29th-Day)
 * @brief example main file
 * @version 0.1
 * @date 2023-04-20
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <stdlib.h>
#include <stdio.h>

#include "engine.h"
#include "display.h"

// #include <chrono>

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
    unsigned int height;
    unsigned int width;
    unsigned int scale;
    unsigned int fps;
    unsigned int seed;
    bool recursive;
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
        else if (EQUAL_S(argv[i], "-r"))
            args->recursive = true;
    }

    assert(args->height > 0, "HEIGHT must be greater than 0");
    assert(args->width > 0, "WIDTH must be greater than 0");
    assert(args->scale > 0, "SCALE must be greater than 0");
    assert(args->fps > 0, "FPS must be greater than 0");
}

int main(int argc, char *argv[])
{
    // SETUP

    Arguments args{ 0 };
    parseArgs(argc, argv, &args);

    args.seed = Engine::InitRandom(args.seed);
    printf("seed: %u\n", args.seed);

    State state;
    Kernel kernel;

    Engine::InitState(&state, args.height, args.width, States::randb);
    Engine::InitKernel(&kernel, Kernels::life);

    // States::Objects::Glider(&state, 50, 40, States::Objects::UP_LEFT);
    // States::Objects::Glider(&state, 50, 50, States::Objects::UP_RIGHT);
    // States::Objects::Glider(&state, 60, 40, States::Objects::DOWN_RIGHT);
    // States::Objects::Glider(&state, 60, 50, States::Objects::DOWN_LEFT);

    // MAIN PART

    Display display = Display(args.height, args.width, args.scale, args.fps);
    while (display.run())
    {
        if (display.nextFrame())
        {
            display.draw(state.current);

            // #ifdef _DEBUG
            // auto t1 = std::chrono::high_resolution_clock::now();
            // #endif _DEBUG

            Engine::Epoch(&state, &kernel, Activations::life, args.recursive);

            // #ifdef _DEBUG
            // auto t2 = std::chrono::high_resolution_clock::now();
            // std::chrono::duration<double, std::milli> ms = t2 - t1;
            // printf("%f ms\n", ms.count());
            // #endif
        }
    }

    Engine::DestroyState(&state);
    Engine::DestroyKernel(&kernel);

    return 0;
}

// int main(int argc, char *argv[])
// {
// }
