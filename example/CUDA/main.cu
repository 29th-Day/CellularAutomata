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
#include <iostream>


using namespace CellularAutomata;

// Custom activation function
ACTIVATION(customActivation,
{
    return (x > 10) ? 0: x;
})

int main(int argc, char* argv[])
{
    const unsigned int HEIGHT = 10;
    const unsigned int WIDTH = 10;
    const bool RECURSIVE = true;
    const unsigned int SEED = 0xAABBCC;

    try
    {
        random::seed(SEED);

        stateFunc<float> sf = nullptr;
        kernelFunc<float> kf = Kernels::life;
        // Activations::life<float> af;
        customActivation<float> af;

        State<float> stateGPU(HEIGHT, WIDTH, sf, Device::CUDA);
        State<float> stateCPU(HEIGHT, WIDTH, sf, Device::CPU);
        Kernel<float> kernel(3, kf, Device::CUDA);

        States::Objects::Glider(&stateCPU, WIDTH / 2, HEIGHT / 2);

        stateCPU.copyTo(&stateGPU);

        for (int i = 0; i < 5; i++)
        {
            stateGPU.copyTo(&stateCPU);
            stateCPU.print();
            std::cout << "-------------------" << std::endl;

            CellularAutomata::Epoch(&stateGPU, &kernel, af, RECURSIVE);
        }
    }
    catch (const std::exception& e)
    {
        std::cout << "Exception occured: " << e.what() << std::endl;
    }

    return 0;
}
