#pragma once

#include "common.hpp"

namespace CellularAutomata
{
    namespace Kernels
    {
        /**
         * @brief Fills the kernel with ones
         *
         * @tparam T type
         * @param array kernel array
         * @param s kernel size
         */
        template <typename T>
        void ones(T* array, const int s);


        /**
         * @brief Fills the kernel with a pattern to run Conway's Game of Life
         *
         * @tparam T type
         * @param array kernel array
         * @param s kernel size
         */
        template <typename T>
        void life(T* array, const int s);

        // template <typename T>
        // void normal(T* array, const int s);
    }
}

#include "kernels.inl"
