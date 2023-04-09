#pragma once

#include "kernels.h"
#include "activations.h"
#include "states.h"


class LifeEngine
{
    public:
        LifeEngine(int height, int width, state_func sf, kernel_func kf, activation_func af);
        ~LifeEngine();

        void epoch();

        float *state;

        int iteration;
        int HEIGHT;
        int WIDTH;
    private:
        float *next;
        float *kernel;

        int KERNEL_SIZE;

        activation_func fn;
};