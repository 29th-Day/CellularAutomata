#include "states.h"
#include "common.h"
#include "rng.h"

#include <stdlib.h>
#include <stdexcept>

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

// States::Objects

inline void copy_into(float *src, int height, int width, State *state, int x, int y)
{
    int i = 0;
    for (int _y = y; _y < y + height; _y++)
    {
        for (int _x = x; _x < x + width; _x++)
        {
            state->current[_y * state->width + _x] = src[i];
            i++;
        }
    }
}

void States::Objects::Glider(State *state, int x, int y, Direction type)
{
    switch (type)
    {
    case Direction::DOWN_RIGHT:
    {
        float glider[3 * 3]{
            0, 1, 0,
            0, 0, 1,
            1, 1, 1};
        copy_into(glider, 3, 3, state, x, y);
        return;
    }
    case Direction::DOWN_LEFT:
    {
        float glider[3 * 3]{
            0, 1, 0,
            1, 0, 0,
            1, 1, 1};
        copy_into(glider, 3, 3, state, x, y);
        return;
    }
    case Direction::UP_RIGHT:
    {
        float glider[3 * 3]{
            1, 1, 1,
            0, 0, 1,
            0, 1, 0};
        copy_into(glider, 3, 3, state, x, y);
        return;
    }
    case Direction::UP_LEFT:
    {
        float glider[3 * 3]{
            1, 1, 1,
            1, 0, 0,
            0, 1, 0};
        copy_into(glider, 3, 3, state, x, y);
        return;
    }
    }
}
