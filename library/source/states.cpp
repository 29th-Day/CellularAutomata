#include "states.h"
#include "common.h"
#include "rng.h"

#include <stdlib.h>

void States::randb(State *state)
{
    for (int i = 0; i < state->height * state->width; i++)
    {
        if (RNG::number())
            state->current[i] = 1.0f;
    }
}

void States::randf(State *state)
{
    for (int i = 0; i < state->height * state->width; i++)
    {
        state->current[i] = RNG::decimal();
    }
}
