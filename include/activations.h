#pragma once

#define EULER_NUMBER_F 2.71828182846f

typedef float (*activation_func)(float);

namespace Activation
{
    float identity(float x);

    float binary(float x);

    float clip(float x);

    float sigmoid(float x);

    float life(float x);

    float linear_3(float x);
}