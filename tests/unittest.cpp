#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "engine.h"

#include <array>

// https://github.com/doctest/doctest

TEST_SUITE("RNG")
{
    TEST_CASE("RNG - specific seed")
    {
        RNG::seed(1);
        float a = RNG::decimal(0.0f, 1.0f);
        float b = RNG::decimal(0.0f, 1.0f);
        RNG::seed(1);
        float c = RNG::decimal(0.0f, 1.0f);
        CHECK(a == c);
        CHECK(a != b);
    }

    TEST_CASE("RNG - random seed")
    {
        RNG::seed(NULL);
        float a = RNG::decimal(0.0f, 1.0f);
        float b = RNG::decimal(0.0f, 1.0f);
        RNG::seed(NULL);
        float c = RNG::decimal(0.0f, 1.0f);
        CHECK(a != b);
        CHECK(a != c);
    }
}

TEST_SUITE("STATES")
{
    TEST_CASE("STATES - init / destroy")
    {
        State state;
        Engine::InitState(&state, 100, 100, NULL);
        CHECK(state.height == 100);
        CHECK(state.width == 100);
        CHECK(state.current != nullptr);
        CHECK(state.next != nullptr);
        CHECK(state.current[0] == 0.0f);

        int MAX = state.height * state.width - 1;
        state.current[MAX] = 2.0f;
        CHECK(state.current[MAX] == 2.0f);

        Engine::DestroyState(&state);
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
        Engine::InitKernel(&kernel, Kernels::full);
        CHECK(kernel.size != 0);
        CHECK(kernel.kernel != nullptr);
        CHECK(kernel.kernel[0] == 1.0f);

        int MAX = kernel.size * kernel.size - 1;
        kernel.kernel[MAX] = 2.0f;
        CHECK(kernel.kernel[MAX] == 2.0f);

        Engine::DestroyKernel(&kernel);
        CHECK(kernel.size == 0);
        CHECK(kernel.kernel == nullptr);
    }

    // test cases for pre-defined kernels
}

TEST_SUITE("ACTIVATIONS")
{
    const int length = 7;
    std::array<float, length> x = {0.0f, 0.5f, 1.0f, 3.1415f, 5.0f, 9.0f, 100.0f};

    TEST_CASE("ACTIVATIONS - identity")
    {
        for (int i = 0; i < length; i++)
        {
            CHECK(Activations::identity(x[i]) == doctest::Approx(x[i]));
        }
    }

    TEST_CASE("ACTIVATIONS - sigmoid")
    {
        std::array<float, length> y = {0.0f, 0.5f, 1.0f, 3.1415f, 5.0f, 9.0f, 100.0f};

        for (int i = 0; i < length; i++)
        {
            CHECK(Activations::identity(x[i]) == doctest::Approx(y[i]));
        }
    }

    TEST_CASE("ACTIVATIONS - life")
    {
    }
}

TEST_SUITE("ENGINE")
{
    TEST_CASE("ENGINE - convolution (center)")
    {
        State state;
        Kernel kernel;

        Engine::InitState(&state, 3, 3, NULL);
        Engine::InitKernel(&kernel, Kernels::full);

        // 0 1 2
        // 3 4 5
        // 6 7 8

        state.current[4] = 1.0f;

        Engine::Epoch(&state, &kernel, Activations::identity);

        for (int i = 0; i < 9; i++)
        {
            CHECK(state.current[i] == doctest::Approx(1.0));
        }

        Engine::DestroyState(&state);
        Engine::DestroyKernel(&kernel);
    }
}
