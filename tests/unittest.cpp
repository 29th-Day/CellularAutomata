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
        float a = RNG::decimal(0, 1);
        float b = RNG::decimal(0, 1);
        RNG::seed(1);
        float c = RNG::decimal(0, 1);
        CHECK(a == c);
        CHECK(a != b);
    }

    TEST_CASE("RNG - random seed")
    {
        RNG::seed(NULL);
        float a = RNG::decimal(0, 1);
        float b = RNG::decimal(0, 1);
        RNG::seed(NULL);
        float c = RNG::decimal(0, 1);
        CHECK(a != b);
        CHECK(a != c);
    }
}

TEST_SUITE("STATES")
{
    TEST_CASE("STATES - init / destroy")
    {
        State state;
        CellularAutomata::InitState(&state, 100, 100, NULL);
        CHECK(state.height == 100);
        CHECK(state.width == 100);
        CHECK(state.current != nullptr);
        CHECK(state.next != nullptr);
        CHECK(state.current[0] == 0.0f);

        int MAX = state.height * state.width - 1;
        state.current[MAX] = 2.0f;
        CHECK(state.current[MAX] == 2.0f);

        CellularAutomata::DestroyState(&state);
        CHECK(state.height == 0);
        CHECK(state.width == 0);
        CHECK(state.current == nullptr);
        CHECK(state.next == nullptr);
    }

    // test cases for pre-defined states
}

TEST_SUITE("KERNELS")
{
    TEST_CASE("KERNELS - init / destroy")
    {
        Kernel kernel;
        CellularAutomata::InitKernel(&kernel, 3, Kernels::full);
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

TEST_SUITE("ACTIVATIONS")
{
    const int length = 10;
    std::array<float, length> x = {-10.0f, -3.0f, -1.0f, 0.0f, 0.5f, 1.0f, 3.1415f, 5.0f, 9.0f, 100.0f};

    TEST_CASE("ACTIVATIONS - sigmoid")
    {
        std::array<float, length> y = {0.00005f, 0.04743f, 0.26894f, 0.5f, 0.62246f, 0.73106f, 0.95857f, 0.99331f, 0.99988f, 1.0f};

        for (int i = 0; i < length; i++)
        {
            CHECK(Activations::sigmoid(x[i]) == doctest::Approx(y[i]));
        }
    }

    TEST_CASE("ACTIVATIONS - life")
    {
        CHECK(Activations::life(0.0f) == 0.0f);
        CHECK(Activations::life(1.0f) == 0.0f);
        CHECK(Activations::life(2.0f) == 0.0f);
        CHECK(Activations::life(2.5f) == 1.0f);
        CHECK(Activations::life(3.0f) == 1.0f);
        CHECK(Activations::life(4.0f) == 0.0f);
        CHECK(Activations::life(5.0f) == 0.0f);
        CHECK(Activations::life(6.0f) == 0.0f);
        CHECK(Activations::life(7.0f) == 0.0f);
        CHECK(Activations::life(8.0f) == 0.0f);
    }
}

TEST_SUITE("ENGINE")
{
    TEST_CASE("ENGINE - convolution (center)")
    {
        State state;
        Kernel kernel;

        CellularAutomata::InitState(&state, 3, 3, NULL);
        CellularAutomata::InitKernel(&kernel, 3, Kernels::full);

        auto activation = [](float x)
        {
            return x;
        };

        // 0 0 0
        // 0 1 0
        // 0 0 0
        state.current[4] = 1.0f;

        const int length = 9;
        std::array<float, 9> result = {
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f};

        CellularAutomata::Epoch(&state, &kernel, activation, false);

        for (int i = 0; i < length; i++)
        {
            INFO("element: ", i);
            CHECK(state.current[i] == doctest::Approx(result[i]));
        }

        CellularAutomata::DestroyState(&state);
        CellularAutomata::DestroyKernel(&kernel);
    }

    TEST_CASE("ENGINE - convolution (corner w/ recursion)")
    {
        State state;
        Kernel kernel;

        CellularAutomata::InitState(&state, 3, 3, NULL);
        CellularAutomata::InitKernel(&kernel, 3, Kernels::full);

        auto activation = [](float x)
        {
            return x;
        };

        // 1 0 0
        // 0 0 0
        // 0 0 0
        state.current[0] = 1.0f;

        const int length = 9;
        std::array<float, 9> result = {
            1.0f, 1.0f, 0.0f,
            1.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f};

        CellularAutomata::Epoch(&state, &kernel, activation, false);

        for (int i = 0; i < length; i++)
        {
            INFO("element: ", i);
            CHECK(state.current[i] == doctest::Approx(result[i]));
        }

        CellularAutomata::DestroyState(&state);
        CellularAutomata::DestroyKernel(&kernel);
    }

    TEST_CASE("ENGINE - convolution (corner w recursion)")
    {
        State state;
        Kernel kernel;

        CellularAutomata::InitState(&state, 3, 3, NULL);
        CellularAutomata::InitKernel(&kernel, 3, Kernels::full);

        auto activation = [](float x)
        {
            return x;
        };

        // 1 0 0
        // 0 0 0
        // 0 0 0
        state.current[0] = 1.0f;

        const int length = 9;
        std::array<float, 9> result = {
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f};

        CellularAutomata::Epoch(&state, &kernel, activation, true);

        for (int i = 0; i < length; i++)
        {
            INFO("element: ", i);
            CHECK(state.current[i] == doctest::Approx(result[i]));
        }

        CellularAutomata::DestroyState(&state);
        CellularAutomata::DestroyKernel(&kernel);
    }
}
