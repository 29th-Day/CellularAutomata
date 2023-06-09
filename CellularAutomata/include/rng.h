#pragma once

#include "common.h"
#include <type_traits>

/**
 * @brief Global Random Number Generator
 */
namespace RNG
{
    /**
     * @brief Seeds the RNG
     *
     * @param seed Randomized if NULL
     * @return actual seed used
     */
    unsigned int seed(unsigned int seed);

    template <typename T>
    integer random(T min, T max);

    template <typename T>
    decimal random(T min, T max);

    template int random(int, int);
    template float random(float, float);

}
