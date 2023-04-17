#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "engine.h"

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
        Engine::DestroyState(&state);
        CHECK(state.height == 0);
        CHECK(state.width == 0);
        CHECK(state.current == nullptr);
        CHECK(state.next == nullptr);
    }
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
        Engine::DestroyKernel(&kernel);
        CHECK(kernel.size == 0);
        CHECK(kernel.kernel == nullptr);
    }
}

TEST_SUITE("ENGINE")
{
    // TODO
}
