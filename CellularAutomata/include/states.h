#pragma once

#include "common.h"

namespace States
{
    template <typename T>
    void normal(T *array, const int h, const int w);

    template <typename T>
    void binary(T *array, const int h, const int w);

    namespace Objects
    {
        enum Direction
        {
            // North
            N,
            // East
            E,
            // South
            S,
            // West
            W
        };

        template <typename T>
        void Glider(State<T> *state, const int x, const int y);

        template <typename T>
        void Spaceship(State<T> *state, const int x, const int y);

        template <typename T>
        void Bipole(State<T> *state, const int x, const int y);

        template <typename T>
        void Tripole(State<T> *state, const int x, const int y);

        template <typename T>
        void f_Pentomino(State<T> *state, const int x, const int y);
    }
}
