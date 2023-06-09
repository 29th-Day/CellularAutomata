#include "states.h"
#include "common.h"
#include "rng.h"

#include <stdexcept>
#include <stdlib.h>

#include <iostream>

template <typename T>
void States::normal(T *array, const int h, const int w)
{
    for (int i = 0; i < h * w; i++)
    {
        array[i] = RNG::random<T>(0, 1);
    }
}

template void States::normal<float>(float *, const int, const int);

template <typename T>
void States::binary(T *array, const int h, const int w)
{
    for (int i = 0; i < h * w; i++)
    {
        if (RNG::random<float>(0, 1) > 0.5)
        {
            array[i] = static_cast<T>(1);
        }
    }
}

template void States::binary<float>(float *, const int, const int);

// States::Objects

template <typename T>
inline void copy_into(int *src, const int src_h, const int src_w, T *dst, const int dst_w, const int y, const int x)
{
    int i = 0;
    for (int _y = y; _y < y + src_h; _y++)
    {
        for (int _x = x; _x < x + src_w; _x++)
        {
            dst[_y * dst_w + _x] = static_cast<T>(src[i]);
            i++;
        }
    }
}

template <typename T>
void States::Objects::Glider(State<T> *state, const int x, const int y)
{
    int pattern[] = {
        0, 1, 0,
        0, 0, 1,
        1, 1, 1};

    copy_into(pattern, 3, 3, state->curr, state->width, y, x);
}

template void States::Objects::Glider<float>(State<float> *, const int, const int);

template <typename T>
void States::Objects::Spaceship(State<T> *state, const int x, const int y)
{
    int pattern[] = {
        0, 1, 1, 1, 1,
        1, 0, 0, 0, 1,
        0, 0, 0, 0, 1,
        1, 0, 0, 1, 0};

    copy_into(pattern, 4, 5, state->curr, state->width, y, x);
}

template void States::Objects::Spaceship<float>(State<float> *, const int, const int);

template <typename T>
void States::Objects::Bipole(State<T> *state, const int x, const int y)
{
    int pattern[] = {
        1, 1, 0, 0,
        1, 0, 0, 0,
        0, 0, 0, 1,
        0, 0, 1, 1};

    copy_into(pattern, 4, 4, state->curr, state->width, y, x);
}

template void States::Objects::Bipole<float>(State<float> *, const int, const int);

template <typename T>
void States::Objects::Tripole(State<T> *state, const int x, const int y)
{
    int pattern[] = {
        1, 1, 0, 0, 0,
        1, 0, 0, 0, 0,
        0, 1, 0, 1, 0,
        0, 0, 0, 0, 1,
        0, 0, 0, 1, 1};

    copy_into(pattern, 5, 5, state->curr, state->width, y, x);
}

template void States::Objects::Tripole<float>(State<float> *, const int, const int);

template <typename T>
void States::Objects::f_Pentomino(State<T> *state, const int x, const int y)
{
    int pattern[] = {
        0, 1, 1,
        1, 1, 0,
        0, 1, 0};

    copy_into(pattern, 3, 3, state->curr, state->width, y, x);
}

template void States::Objects::f_Pentomino<float>(State<float> *, const int, const int);
