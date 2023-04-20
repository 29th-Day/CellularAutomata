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
            DOWN_RIGHT,
            DOWN_LEFT,
            UP_RIGHT,
            UP_LEFT
        };

        void Glider(State *state, int x, int y, Direction type);
    }
}
