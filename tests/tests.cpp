// https://github.com/doctest/doctest
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <CellularAutomata>
#include <array>

#include "shared.hpp"

using namespace CellularAutomata;

/*
TEST_SUITE("RNG")
{
    TEST_CASE("RNG - specific seed")
    {
        RNG::seed(1);
        float a = RNG::random<float>(0, 1);
        float b = RNG::random<float>(0, 1);
        RNG::seed(1);
        float c = RNG::random<float>(0, 1);
        CHECK(a == c);
        CHECK(a != b);
    }

    TEST_CASE("RNG - random seed")
    {
        RNG::seed(NULL);
        float a = RNG::random<float>(0, 1);
        float b = RNG::random<float>(0, 1);
        RNG::seed(NULL);
        float c = RNG::random<float>(0, 1);
        CHECK(a != b);
        CHECK(a != c);
    }
}
*/

TEST_SUITE("STATES")
{
    const int h = 10;
    const int w = 10;
    const stateFunc<int> f = nullptr;

    TEST_CASE("CORE - states/constructor")
    {
        State<int> state(h, w, f, Device::CPU);
        CHECK(state.height == h);
        CHECK(state.width == w);
        CHECK(state.curr != nullptr);
        CHECK(state.next != nullptr);
        CHECK(state.curr[0] == 0);

        const int MAX = h * w - 1;
        state.curr[MAX] = 2;
        CHECK(state.curr[MAX] == 2);
    }
}

TEST_SUITE("KERNELS")
{
    const int s = 3;
    const kernelFunc<int> f = nullptr;

    TEST_CASE("CORE - kernels/constructor")
    {
        Kernel<int> kernel(s, f, Device::CPU);
        CHECK(kernel.size == s);
        CHECK(kernel.kernel != nullptr);
        CHECK(kernel.kernel[0] == 0);

        const int MAX = s * s - 1;
        kernel.kernel[MAX] = 2;
        CHECK(kernel.kernel[MAX] == 2);
    }
}

TEST_SUITE("CONVOLUTION")
{
    const int h = 5;
    const int w = 5;
    const int s = 3;
    const int length = h * w;

    TEST_CASE_TEMPLATE("CORE - conv/center", T, types)
    {
        State<T> state(h, w, (stateFunc<T>)nullptr, Device::CPU);
        Kernel<T> kernel(s, Kernels::ones, Device::CPU);
        Activations::normal<T> fn;

        // 0 0 0 0 0
        // 0 0 0 0 0
        // 0 0 1 0 0
        // 0 0 0 0 0
        // 0 0 0 0 0
        state.curr[12] = 1;

        std::array<T, length> result = {
            0, 0, 0, 0, 0,
            0, 1, 1, 1, 0,
            0, 1, 1, 1, 0,
            0, 1, 1, 1, 0,
            0, 0, 0, 0, 0 };

        SUBCASE("non-recursive")
        {
            CellularAutomata::Epoch(&state, &kernel, fn, false);

            CHECK_ARRAY(state.curr, result, length);
        }

        SUBCASE("recursive")
        {
            CellularAutomata::Epoch(&state, &kernel, fn, true);

            CHECK_ARRAY(state.curr, result, length);
        }
    }

    TEST_CASE_TEMPLATE("CORE - conv/corner", T, types)
    {
        State<T> state(h, w, (stateFunc<T>)nullptr, Device::CPU);
        Kernel<T> kernel(s, Kernels::ones, Device::CPU);
        Activations::normal<T> fn;

        // 1 0 0 0 0
        // 0 0 0 0 0
        // 0 0 0 0 0
        // 0 0 0 0 0
        // 0 0 0 0 0
        state.curr[0] = 1;

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

            CHECK_ARRAY(state.curr, result, length);
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

            CHECK_ARRAY(state.curr, result, length);
        }
    }
}
