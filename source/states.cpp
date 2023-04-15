#include "states.h"
#include "common.h"

#include <stdlib.h>

inline float random()
{
    return (float)rand() / (float)(RAND_MAX);
}

void States::randb(State *state)
{
    for (int i = 0; i < state->height * state->width; i++)
    {
        if (rand() % 2)
            state->current[i] = 1.0f;
    }
}

void States::randf(State *state)
{
    for (int i = 0; i < state->height * state->width; i++)
    {
        state->current[i] = random();
    }
}
