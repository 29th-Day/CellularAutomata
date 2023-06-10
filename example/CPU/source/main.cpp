/**
 * @file main.cpp
 * @author 29th-Day (https://github.com/29th-Day)
 * @brief Example for CPU backend
 * @version 0.1
 * @date 2023-06-10
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "CellularAutomata.h"

#include "display.h"
#include "timeit.h"

int main(int argc, char *argv[])
{
    const int HEIGHT = 100;
    const int WIDTH = 100;
    const int SCALE = 4;
    const int FPS = 30;
    const bool RECURSIVE = true;
    const unsigned int SEED = 0xAABBCC;

    CellularAutomata::InitRandom(SEED);

    State<float> state;
    Kernel<float> kernel;

    stateFunc<float> sf = States::normal;
    kernelFunc<float> kf = Kernels::life;
    Activations::OpCode af = Activations::life;

    CellularAutomata::InitState(&state, HEIGHT, WIDTH, sf, Device::CPU);
    CellularAutomata::InitKernel(&kernel, 3, kf, Device::CPU);

    Display display = Display(HEIGHT, WIDTH, SCALE, FPS);
    while (display.run())
    {
        if (display.nextFrame())
        {
            display.draw(state.curr);

            timeit(CellularAutomata::Epoch(&state, &kernel, af, RECURSIVE));
        }
    }

    CellularAutomata::DestroyState(&state);
    CellularAutomata::DestroyKernel(&kernel);

    return 0;
}
