#include "activations.h"

#include <cmath>

#define EULER_NUMBER_F 2.71828182846f

template <typename T>
cudaFn T Activations::_normal(T x)
{
    if (x < 0)
        return static_cast<T>(0);
    else if (x > 1)
        return static_cast<T>(1);
    else
        return x;
}

template cudaFn float Activations::_normal(float);

template <typename T>
cudaFn T Activations::_life(T x)
{
    unsigned char u = static_cast<unsigned char>(x);

    unsigned char neighbours = u & 0xF; // low  nibble
    bool alive = u & 0xF0;              // high nibble

    switch (neighbours)
    {
    case 2:
        // staying alive
        return static_cast<T>((alive) ? 1 : 0);
    case 3:
        // birth
        return static_cast<T>(1);
    default:
        // under- / overpopulation
        return static_cast<T>(0);
    }
}

template cudaFn float Activations::_life(float);

template <typename T>
cudaFn T Activations::_sigmoid(T x)
{
    return (1 / (1 + std::pow(EULER_NUMBER_F, -x)));
}

template cudaFn float Activations::_sigmoid(float);

template <typename T>
cudaFn T Activations::_tanh(T x)
{
    return std::tanh(x);
}

template cudaFn float Activations::_tanh(float);
