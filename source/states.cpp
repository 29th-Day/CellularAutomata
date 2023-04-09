#include "states.h"

#include <stdlib.h>

inline float random()
{
    return (float)rand()/(float)(RAND_MAX);
}

void State::randb(float current[], int height, int width)
{
    for (int i = 0; i < height * width; i++)
    {
        if (rand() % 2)
            current[i] = 1.0f;
    }
}

void State::randf(float current[], int height, int width)
{
    for (int i = 0; i < height * width; i++)
    {
        current[i] = random();
    }
}
