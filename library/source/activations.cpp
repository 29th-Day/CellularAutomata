#include "activations.h"

#include <cmath>

#define EULER_NUMBER_F 2.71828182846f

float Activations::identity(float x)
{
    return x;
}

float Activations::sigmoid(float x)
{
    return (1 / (1 + std::powf(EULER_NUMBER_F, -x)));
}

float Activations::life(float x)
{
    int neighbors = (int)x;
    bool alive = ((x - neighbors) > 0) ? true : false;

    // if (neighbors == 3)
    //     // birth
    //     return 1.0f;
    // else if (neighbors < 2)
    //     // loneliness
    //     return 0.0f;
    // else if (neighbors > 3)
    //     // over population
    //     return 0.0f;
    // else if (alive && neighbors == 2)
    //     // staying alive
    //     return 1.0f;
    // else
    //     return 0.0f;

    switch (neighbors)
    {
    case 2:
        // staying alive
        return (alive) ? 1.0f : 0.0f;
    case 3:
        // birth
        return 1.0f;
    default:
        // under- / overpopulation
        return 0.0f;
    }
}

float Activations::clip(float x)
{
    if (x < 0.0f)
        return 0.0f;
    else if (x > 1.0f)
        return 1.0f;
    else
        return x;
}

float Activations::sin(float x)
{
    return std::sin(x);
}

float Activations::cos(float x)
{
    return std::cos(x);
}

float Activations::tan(float x)
{
    return std::tan(x);
}
