#include "activations.h"

#include <math.h>

float Activation::identity(float x)
{
    return x;
}

float Activation::sigmoid(float x)
{
    return (1 / (1 + powf(EULER_NUMBER_F, -x)));
}

float Activation::life(float x)
{
    int neighbors = (int)x;
    bool alive = ((x - neighbors) > 0) ? true : false;

    if (neighbors == 3)
        // birth
        return 1.0f;
    else if (neighbors < 2)
        // loneliness
        return 0.0f;
    else if (neighbors > 3)
        // over population
        return 0.0f;
    else if (alive && neighbors == 2)
        // staying alife
        return 1.0f;
    else
        return 0.0f;
}