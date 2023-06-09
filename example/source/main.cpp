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

#include "CellularAutomata.h"

#include "display.h"

#include <iomanip>
#include <iostream>
#include <string>

#include <chrono>

#include "backend_cuda.h"

#define EQUAL_S(a, b) strcmp(a, b) == 0

#define assert(x, msg)                                   \
    {                                                    \
        if (!(x))                                        \
        {                                                \
            fprintf(stderr, "Assertion error: %s", msg); \
            exit(-2);                                    \
        }                                                \
    }

#define timeit(fn)                                          \
    auto t1 = std::chrono::high_resolution_clock::now();    \
    fn;                                                     \
    auto t2 = std::chrono::high_resolution_clock::now();    \
    std::chrono::duration<double, std::milli> ms = t2 - t1; \
    std::cout << ms.count() << " ms" << std::endl;

template <typename T>
void print2D(T *array, const int height, const int width)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            std::cout << std::setfill(' ') << std::setw(5) << std::fixed << std::setprecision(2) << array[y * width + x] << " ";
        }
        std::cout << std::endl;
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

#define CENTER_X args.width / 2
#define CENTER_Y args.height / 2

int main(int argc, char *argv[])
{
    // SETUP

    // passLambda(Activations::lambda<float>);

    // return 0;

    Arguments args{0};
    parseArgs(argc, argv, &args);

    args.seed = CellularAutomata::InitRandom(args.seed);

    printf("seed: %8X\n", args.seed);

    State<float> stateGPU;
    State<float> stateCPU;
    Kernel<float> kernel;

    stateFunc<float> sf = nullptr;
    kernelFunc<float> kf = Kernels::life;
    Activations::OpCode af = Activations::life;

    CellularAutomata::InitState(&stateGPU, args.height, args.width, sf, Device::CUDA);
    CellularAutomata::InitState(&stateCPU, args.height, args.width, (stateFunc<float>)nullptr, Device::CPU);

    CellularAutomata::InitKernel(&kernel, 3, kf, Device::CUDA);

    // print2D(state.curr, state.height, state.width);
    // print2D(kernel.kernel, kernel.size, kernel.size);

    // States::Objects::Glider(&stateCPU, 90, 10);
    // States::Objects::Spaceship(&stateCPU, 80, 70);
    // States::Objects::Bipole(&stateCPU, 60, 60);
    // States::Objects::Tripole(&stateCPU, 10, 80);
    States::Objects::f_Pentomino(&stateCPU, CENTER_X, CENTER_Y);

    CellularAutomata::CopyTo(&stateCPU, &stateGPU);

    // print2D(stateCPU.curr, stateCPU.height, stateCPU.width);

    // std::cout << std::setfill('-') << std::setw(100) << "" << std::endl;

    // CellularAutomata::Epoch(&stateGPU, &kernel, af, args.recursive);

    // CellularAutomata::CopyTo(&stateGPU, &stateCPU);

    // print2D(stateCPU.curr, stateCPU.height, stateCPU.width);

    // MAIN PART

    Display display = Display(args.height, args.width, args.scale, args.fps);
    while (display.run())
    {
        if (display.nextFrame())
        {
            CellularAutomata::CopyTo(&stateGPU, &stateCPU);
            display.draw(stateCPU.curr);

            timeit(CellularAutomata::Epoch(&stateGPU, &kernel, af, args.recursive));
            // CellularAutomata::Epoch(&stateGPU, &kernel, af, args.recursive);
        }
    }

    CellularAutomata::DestroyState(&stateGPU);
    CellularAutomata::DestroyState(&stateCPU);
    CellularAutomata::DestroyKernel(&kernel);

    return 0;
}
