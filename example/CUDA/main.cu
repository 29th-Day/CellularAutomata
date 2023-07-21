/**
 * @file main.cpp
 * @author 29th-Day (https://github.com/29th-Day)
 * @brief Example for CUDA backend
 * @version 0.1
 * @date 2023-07-20
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <CellularAutomata>

using namespace CellularAutomata;

// Custom activation function
ACTIVATION(identity,
{
    return x;
})

int main(int argc, char* argv[])
{
    const int HEIGHT = 10;
    const int WIDTH = 10;
    const bool RECURSIVE = true;
    const unsigned int SEED = 0xAABBCC;

    random::seed(SEED);

    stateFunc<float> sf = nullptr;
    kernelFunc<float> kf = Kernels::life;
    // Activations::life<float> af;
    identity<float> af;

    State<float> stateGPU(HEIGHT, WIDTH, sf, Device::CUDA);
    State<float> stateCPU(HEIGHT, WIDTH, sf, Device::CPU);
    Kernel<float> kernel(3, kf, Device::CUDA);


    States::Objects::Bipole1(&stateCPU, WIDTH / 2, HEIGHT / 2);

    stateCPU.copyTo(&stateGPU);

    for (int i = 0; i < 5; i++)
    {
        stateGPU.copyTo(&stateCPU);
        stateCPU.print();
        std::cout << "-----" << std::endl;

        CellularAutomata::Epoch(&stateGPU, &kernel, af, RECURSIVE);
    }

    return 0;
}
