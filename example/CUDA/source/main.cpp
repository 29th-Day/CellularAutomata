/**
 * @file main.cpp
 * @author 29th-Day (https://github.com/29th-Day)
 * @brief Example for CUDA backend
 * @version 0.1
 * @date 2023-06-10
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "CellularAutomata.h"

int main(int argc, char *argv[])
{
    const int HEIGHT = 100;
    const int WIDTH = 100;
    const int SCALE = 4;
    const int FPS = 60;
    const bool RECURSIVE = true;
    const unsigned int SEED = 0xAABBCC;

    CellularAutomata::InitRandom(SEED);

    State<float> stateGPU;
    State<float> stateCPU;
    Kernel<float> kernel;

    stateFunc<float> sf = nullptr;
    kernelFunc<float> kf = Kernels::life;
    Activations::OpCode af = Activations::life;

    CellularAutomata::InitState(&stateGPU, HEIGHT, WIDTH, sf, Device::CUDA);
    CellularAutomata::InitState(&stateCPU, HEIGHT, WIDTH, sf, Device::CPU);

    CellularAutomata::InitKernel(&kernel, 3, kf, Device::CPU);

    // States::Objects::Glider(&stateCPU, 50, 50);
    // States::Objects::Spaceship(&stateCPU, 50, 50);
    // States::Objects::Bipole(&stateCPU, 50, 50);
    // States::Objects::Tripole(&stateCPU, 50, 50);
    States::Objects::f_Pentomino(&stateCPU, WIDTH / 2, HEIGHT / 2);

    CellularAutomata::CopyTo(&stateCPU, &stateGPU);

    for (int i = 0; i < 5; i++)
    {
        CellularAutomata::Epoch(&stateGPU, &kernel, af, RECURSIVE);

        CellularAutomata::CopyTo(&stateGPU, &stateCPU);
        // use stateCPU.curr
    }

    CellularAutomata::DestroyState(&stateGPU);
    CellularAutomata::DestroyState(&stateCPU);
    CellularAutomata::DestroyKernel(&kernel);

    return 0;
}
