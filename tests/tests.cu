// https://github.com/doctest/doctest
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <CellularAutomata>
#include <array>

#include "shared.hpp"

using namespace CellularAutomata;

TEST_SUITE("CONVOLUTION")
{
    const int h = 5;
    const int w = 5;
    const int s = 3;
    const int length = h * w;

    TEST_CASE("conv/center (CUDA)")
    {
        State<int> temp(h, w, (stateFunc<int>)nullptr, Device::CPU);
        State<int> state(h, w, (stateFunc<int>)nullptr, Device::CUDA);
        Kernel<int> kernel(s, Kernels::ones, Device::CUDA);
        Activations::normal<int> fn;

        // 0 0 0 0 0
        // 0 0 0 0 0
        // 0 0 1 0 0
        // 0 0 0 0 0
        // 0 0 0 0 0
        temp.curr[12] = 1;
        temp.copyTo(&state);

        std::array<int, length> result = {
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

    TEST_CASE("conv/corner (CUDA)")
    {
        State<int> temp(h, w, (stateFunc<int>)nullptr, Device::CPU);
        State<int> state(h, w, (stateFunc<int>)nullptr, Device::CUDA);
        Kernel<int> kernel(s, Kernels::ones, Device::CUDA);
        Activations::normal<int> fn;

        // 1 0 0 0 0
        // 0 0 0 0 0
        // 0 0 0 0 0
        // 0 0 0 0 0
        // 0 0 0 0 0
        temp.curr[0] = 1;
        temp.copyTo(&state);

        SUBCASE("non-recursive")
        {
            std::array<int, length> result = {
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
            std::array<int, length> result = {
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
