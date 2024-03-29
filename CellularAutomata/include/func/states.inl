#include "states.hpp"
#include "../core/rng.hpp"
#include "../core/exceptions.hpp"

namespace CellularAutomata
{
    namespace States
    {
        template <typename T>
        void normal(T* array, const unsigned int h, const unsigned int w)
        {
            for (int i = 0; i < h * w; i++)
            {
                array[i] = CellularAutomata::random::random<T>(0, 1);
            }
        }

        template <typename T>
        void binary(T* array, const unsigned int h, const unsigned int w)
        {
            for (int i = 0; i < h * w; i++)
            {
                if (CellularAutomata::random::random<float>(0, 1) > 0.5)
                {
                    array[i] = static_cast<T>(1);
                }
                else
                {
                    array[i] = static_cast<T>(0);
                }
            }
        }

        namespace Objects
        {
            template <typename T>
            inline void copyInto(T* src, const unsigned int src_h, const unsigned int src_w, T* dst, const unsigned int dst_h, const unsigned int dst_w, const unsigned int y, const unsigned int x)
            {
                if (src_h + y > dst_h || src_w + x > dst_w)
                    throw exception::OutOfBounds();

                int i = 0;
                for (int _y = y; _y < y + src_h; _y++)
                {
                    for (int _x = x; _x < x + src_w; _x++)
                    {
                        dst[_y * dst_w + _x] = src[i];
                        i++;
                    }
                }
            }

            template <typename T>
            void Glider(State<T>* state, const unsigned int x, const unsigned int y)
            {
                T pattern[] = {
                    0, 1, 0,
                    0, 0, 1,
                    1, 1, 1 };

                copyInto(pattern, 3, 3, state->curr, state->height, state->width, y, x);
            }

            template <typename T>
            void Spaceship(State<T>* state, const unsigned int x, const unsigned int y)
            {
                T pattern[] = {
                    0, 1, 1, 1, 1,
                    1, 0, 0, 0, 1,
                    0, 0, 0, 0, 1,
                    1, 0, 0, 1, 0 };

                copyInto(pattern, 4, 5, state->curr, state->height, state->width, y, x);
            }

            template <typename T>
            void Bipole1(State<T>* state, const unsigned int x, const unsigned int y)
            {
                T pattern[] = {
                    1, 1, 0, 0,
                    1, 0, 0, 0,
                    0, 0, 0, 1,
                    0, 0, 1, 1 };

                copyInto(pattern, 4, 4, state->curr, state->height, state->width, y, x);
            }

            template <typename T>
            void Bipole2(State<T>* state, const unsigned int x, const unsigned int y)
            {
                T pattern[] = {
                    1, 1, 0, 0, 0,
                    1, 0, 0, 0, 0,
                    0, 1, 0, 1, 0,
                    0, 0, 0, 0, 1,
                    0, 0, 0, 1, 1 };

                copyInto(pattern, 5, 5, state->curr, state->height, state->width, y, x);
            }

            template <typename T>
            void r_Pentomino(State<T>* state, const unsigned int x, const unsigned int y)
            {
                T pattern[] = {
                    0, 1, 1,
                    1, 1, 0,
                    0, 1, 0 };

                copyInto(pattern, 3, 3, state->curr, state->height, state->width, y, x);
            }
        }
    }
}
