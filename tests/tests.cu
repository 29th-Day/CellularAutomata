// https://github.com/doctest/doctest
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <CellularAutomata>
#include <array>

#include "shared.hpp"

using namespace CellularAutomata;

#define _types int, float

TEST_SUITE("CONVOLUTION")
{
    const int h = 5;
    const int w = 5;
    const int s = 3;
    const int length = h * w;

    TEST_CASE_TEMPLATE("CUDA - conv/center", T, _types)
    {
        State<T> temp(h, w, (stateFunc<T>)nullptr, Device::CPU);
        State<T> state(h, w, (stateFunc<T>)nullptr, Device::CUDA);
        Kernel<T> kernel(s, Kernels::ones, Device::CUDA);
        Activations::normal<T> fn;

        // 0 0 0 0 0
        // 0 0 0 0 0
        // 0 0 1 0 0
        // 0 0 0 0 0
        // 0 0 0 0 0
        temp.curr[12] = 1;
        temp.copyTo(&state);

        std::array<T, length> result = {
            0, 0, 0, 0, 0,
            0, 1, 1, 1, 0,
            0, 1, 1, 1, 0,
            0, 1, 1, 1, 0,
            0, 0, 0, 0, 0 };

        SUBCASE("non-recursive")
        {
            CellularAutomata::Epoch(&state, &kernel, fn, false);

            state.copyTo(&temp);
            CHECK_ARRAY(temp.curr, result, length);
        }

        SUBCASE("recursive")
        {
            CellularAutomata::Epoch(&state, &kernel, fn, true);

            state.copyTo(&temp);
            CHECK_ARRAY(temp.curr, result, length);
        }
    }

    TEST_CASE_TEMPLATE("CUDA - conv/corner", T, _types)
    {
        State<T> temp(h, w, (stateFunc<T>)nullptr, Device::CPU);
        State<T> state(h, w, (stateFunc<T>)nullptr, Device::CUDA);
        Kernel<T> kernel(s, Kernels::ones, Device::CUDA);
        Activations::normal<T> fn;

        // 1 0 0 0 0
        // 0 0 0 0 0
        // 0 0 0 0 0
        // 0 0 0 0 0
        // 0 0 0 0 0
        temp.curr[0] = 1;
        temp.copyTo(&state);

        std::array<T, length> result;

        SUBCASE("non-recursive")
        {
            result = {
                1, 1, 0, 0, 0,
                1, 1, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0 };

            CellularAutomata::Epoch(&state, &kernel, fn, false);

            state.copyTo(&temp);
            CHECK_ARRAY(temp.curr, result, length);
        }

        SUBCASE("recursive")
        {
            result = {
                1, 1, 0, 0, 1,
                1, 1, 0, 0, 1,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                1, 1, 0, 0, 1 };

            CellularAutomata::Epoch(&state, &kernel, fn, true);

            state.copyTo(&temp);
            CHECK_ARRAY(temp.curr, result, length);
        }
    }

}
