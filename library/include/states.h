#pragma once

#include "common.h"

namespace States
{
    void randb(State *state);

    void randf(State *state);

    namespace Objects
    {
        enum Direction
        {
            // North / Up
            N,
            // North east / Up right
            NE,
            // East / Right
            E,
            // South east / Down right
            SE,
            // South / Down
            S,
            // South west / Down left
            SW,
            // West / Left
            W,
            // North west / Up left
            NW
        };

        void Glider(State *state, int x, int y, Direction type);
    }
}
