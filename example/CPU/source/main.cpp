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

#include "CellularAutomata.hpp"

#include "display.h"
#include "timeit.h"

using namespace CellularAutomata;

int main(int argc, char* argv[])
{
    const int HEIGHT = 100;
    const int WIDTH = 100;
    const int SCALE = 4;
    const int FPS = 30;
    const bool RECURSIVE = true;
    const unsigned int SEED = 0xAABBCC;

    random::seed(SEED);

    stateFunc<float>         sf = nullptr; // States::normal
    kernelFunc<float>        kf = Kernels::life;
    Activations::life<float> af;

    State<float> state(HEIGHT, WIDTH, sf, Device::CPU);
    Kernel<float> kernel(3, kf, Device::CPU);

    States::Objects::Glider(&state, 5, 5);

    Display display = Display(HEIGHT, WIDTH, SCALE, FPS);
    while (display.run())
    {
        if (display.nextFrame())
        {
            display.draw(state.curr);

            timeit(Epoch(&state, &kernel, af, RECURSIVE));
        }
    }

    return 0;
}
