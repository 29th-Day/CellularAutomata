#pragma once

#include "common.hpp"

namespace CellularAutomata
{
    namespace cpu
    {
        /**
         * @brief Runs a single epoch on the CPU.
         * @attention Not not be called manually
         * @note Automatically utilizes OpenMP if linked
         *
         * @tparam T type
         * @tparam Activation functor
         * @param input current state array
         * @param kernel kernel for convolution
         * @param output next state array
         * @param fn activation function
         * @param h height of state array
         * @param w width of state array
         * @param s size of kernel
         * @param r grid recursion enabled
         */
        template <typename T, typename Activation>
        void epoch(
            const T* input, const T* kernel, T* output, Activation fn,
            const unsigned int h, const unsigned int w, const unsigned int s, const bool r);
    }
}

#include "cpu.inl"
