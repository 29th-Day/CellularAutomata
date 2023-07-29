#pragma once

#include "../core/common.hpp"

namespace CellularAutomata
{
    namespace Kernels
    {
        template <typename T>
        void ones(T* array, const int s)
        {
            for (int i = 0; i < s * s; i++)
            {
                array[i] = static_cast<T>(1);
            }
        }

        template <typename T>
        void life(T* array, const int s)
        {
            int center = (s * s) / 2;
            for (int i = 0; i < s * s; i++)
            {
                if (i == center)
                    array[i] = static_cast<T>(16);
                else
                    array[i] = static_cast<T>(1);
            }
        }

        // template <typename T>
        // void normal(T* array, const int s);
    }
}
