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

#include "CellularAutomata.h"
#include "display.h"

#include <string>

#define TIME

#ifdef TIME
#include <chrono>
#endif

#define assert(x, msg)                                   \
    {                                                    \
        if (!(x))                                        \
        {                                                \
            fprintf(stderr, "Assertion error: %s", msg); \
            exit(-2);                                    \
        }                                                \
    }

#define EQUAL_S(a, b) strcmp(a, b) == 0

void print2D(float *array, int height, int width)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            printf("%4.1f ", array[y * width + x]);
        }
        printf("\n");
    }
}

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
    for (int i = 1; i < argc; i++)
    {
        if (EQUAL_S(argv[i], "-h"))
            args->height = std::stoi(argv[++i]);
        else if (EQUAL_S(argv[i], "-w"))
            args->width = std::stoi(argv[++i]);
        else if (EQUAL_S(argv[i], "-s"))
            args->scale = std::stoi(argv[++i]);
        else if (EQUAL_S(argv[i], "-fps"))
            args->fps = std::stoi(argv[++i]);
        else if (EQUAL_S(argv[i], "-seed"))
            args->seed = std::stoul(argv[++i]);
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

    Arguments args{0};
    parseArgs(argc, argv, &args);

    args.seed = CellularAutomata::InitRandom(args.seed);
    printf("seed: %u\n", args.seed);

    State state;
    Kernel kernel;

    CellularAutomata::InitState(&state, args.height, args.width, NULL);
    CellularAutomata::InitKernel(&kernel, 3, Kernels::life);

    print2D(kernel.kernel, kernel.size, kernel.size);

    States::Objects::Glider(&state, 50, 40, States::Objects::NW);
    States::Objects::Glider(&state, 50, 50, States::Objects::SW);
    States::Objects::Glider(&state, 60, 40, States::Objects::NE);
    States::Objects::Glider(&state, 60, 50, States::Objects::SE);

    // MAIN PART

    Display display = Display(args.height, args.width, args.scale, args.fps);
    while (display.run())
    {
        if (display.nextFrame())
        {
            display.draw(state.current);

#ifdef TIME
            auto t1 = std::chrono::high_resolution_clock::now();
#endif

            CellularAutomata::Epoch(&state, &kernel, Activations::life, args.recursive);

#ifdef TIME
            auto t2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> ms = t2 - t1;
            printf("%f ms\n", ms.count());
#endif
        }
    }

    // for (int i = 0; i < 10000; i++)
    // {
    //     CellularAutomata::Epoch(&state, &kernel, Activations::life, args.recursive);
    // }

    CellularAutomata::DestroyState(&state);
    CellularAutomata::DestroyKernel(&kernel);

    return 0;
}
