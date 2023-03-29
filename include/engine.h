#pragma once

class LifeEngine
{
    public:
        LifeEngine();
        ~LifeEngine();

        void epoch();

    private:
        void convolution();

        float *current;
        float *next;
        float *kernel;

        int HEIGHT;
        int WIDTH;
        int KERNEL_SIZE;
};