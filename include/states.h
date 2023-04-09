#pragma once

typedef void (*state_func)(float[], int, int);

namespace State
{
    void randb(float current[], int height, int width);

    void randf(float current[], int height, int width);
}