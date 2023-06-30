#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "CellularAutomata.h"

#include <array>

// https://github.com/doctest/doctest

// #define CHECK_ARRAY_APPROX(a, b, l)               \
//     {                                             \
//         for (int i = 0; i < l; i++)               \
//         {                                         \
//             CHECK(a[i] == doctest::Approx(b[i])); \
//         }                                         \
//     }

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

TEST_SUITE("STATES")
{
    TEST_CASE("STATES - init / destroy [CPU]")
    {
        State<float> state;
        CellularAutomata::InitState(&state, 100, 100, (stateFunc<float>)nullptr, Device::CPU);
        CHECK(state.height == 100);
        CHECK(state.width == 100);
        CHECK(state.curr != nullptr);
        CHECK(state.next != nullptr);
        CHECK(state.curr[0] == 0.0f);

        int MAX = state.height * state.width - 1;
        state.curr[MAX] = 2.0f;
        CHECK(state.curr[MAX] == 2.0f);

        CellularAutomata::DestroyState(&state);
        CHECK(state.height == 0);
        CHECK(state.width == 0);
        CHECK(state.curr == nullptr);
        CHECK(state.next == nullptr);
    }

    // test cases for pre-defined states
}

TEST_SUITE("KERNELS")
{
    TEST_CASE("KERNELS - init / destroy [CPU]")
    {
        Kernel<float> kernel;
        CellularAutomata::InitKernel(&kernel, 3, Kernels::ones, Device::CPU);
        CHECK(kernel.size != 0);
        CHECK(kernel.kernel != nullptr);
        CHECK(kernel.kernel[0] == 1.0f);

        int MAX = kernel.size * kernel.size - 1;
        kernel.kernel[MAX] = 2.0f;
        CHECK(kernel.kernel[MAX] == 2.0f);

        CellularAutomata::DestroyKernel(&kernel);
        CHECK(kernel.size == 0);
        CHECK(kernel.kernel == nullptr);
    }

    // test cases for pre-defined kernels
}

TEST_SUITE("ENGINE")
{
    TEST_CASE("ENGINE - convolution (center) [CPU]")
    {
        State<float> state;
        Kernel<float> kernel;

        CellularAutomata::InitState(&state, 3, 3, (stateFunc<float>)nullptr, Device::CPU);
        CellularAutomata::InitKernel(&kernel, 3, Kernels::ones, Device::CPU);

        // auto activation = [](float x)
        // {
        //     return x;
        // };

        // 0 0 0
        // 0 1 0
        // 0 0 0
        state.curr[4] = 1.0f;

        const int length = 9;
        std::array<float, length> result = {
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f};

        CellularAutomata::Epoch(&state, &kernel, Activations::normal, false);

        for (int i = 0; i < length; i++)
        {
            INFO("element: ", i);
            CHECK(state.curr[i] == doctest::Approx(result[i]));
        }

        CellularAutomata::DestroyState(&state);
        CellularAutomata::DestroyKernel(&kernel);
    }

    TEST_CASE("ENGINE - convolution (corner w/ recursion) [CPU]")
    {
        State<float> state;
        Kernel<float> kernel;

        CellularAutomata::InitState(&state, 3, 3, (stateFunc<float>)nullptr, Device::CPU);
        CellularAutomata::InitKernel(&kernel, 3, Kernels::ones, Device::CPU);

        // auto activation = [](float x)
        // {
        //     return x;
        // };

        // 1 0 0
        // 0 0 0
        // 0 0 0
        state.curr[0] = 1.0f;

        const int length = 9;
        std::array<float, length> result = {
            1.0f, 1.0f, 0.0f,
            1.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f};

        CellularAutomata::Epoch(&state, &kernel, Activations::normal, false);

        for (int i = 0; i < length; i++)
        {
            INFO("element: ", i);
            CHECK(state.curr[i] == doctest::Approx(result[i]));
        }

        CellularAutomata::DestroyState(&state);
        CellularAutomata::DestroyKernel(&kernel);
    }

    TEST_CASE("ENGINE - convolution (corner w recursion) [CPU]")
    {
        State<float> state;
        Kernel<float> kernel;

        CellularAutomata::InitState(&state, 3, 3, (stateFunc<float>)nullptr, Device::CPU);
        CellularAutomata::InitKernel(&kernel, 3, Kernels::ones, Device::CPU);

        // auto activation = [](float x)
        // {
        //     return x;
        // };

        // 1 0 0
        // 0 0 0
        // 0 0 0
        state.curr[0] = 1.0f;

        const int length = 9;
        std::array<float, length> result = {
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f};

        CellularAutomata::Epoch(&state, &kernel, Activations::normal, true);

        for (int i = 0; i < length; i++)
        {
            INFO("element: ", i);
            CHECK(state.curr[i] == doctest::Approx(result[i]));
        }

        CellularAutomata::DestroyState(&state);
        CellularAutomata::DestroyKernel(&kernel);
    }
}
