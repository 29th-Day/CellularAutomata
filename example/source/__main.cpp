#include "test.h"

void fn()
{
    // do stuff
}

int main()
{
    InitRandom(1);

    State<float> state;
    Kernel<float> kernel;

    InitState(&state, 1, 1, fn, Device::CPU);
    InitKernel(&kernel, 1, fn, Device::CPU);

    CopyTo(&state, &state);

    Epoch(&state, &kernel, fn, false);

    DestroyState(&state);
    DestroyKernel(&kernel);
}
