#pragma once

#include "common.hpp"

#include <type_traits>

// float, double, long double
#define decimal typename std::enable_if<std::is_floating_point<T>::value, T>::type

// bool, char, wchar, short, int, long, long long (or extended)
#define integer typename std::enable_if<std::is_integral<T>::value, T>::type

namespace CellularAutomata
{
    /**
     * @brief Global random number generator wrapping the std::random utilities
     * @attention Since `random` works using *static* variables, each translation unit will have a different RNG state
     * @note The first seed used before calling `init()` is random
     */
    namespace random
    {
        /**
         * @brief Sets a random seed for the random number generator
         *
         * @return seed used
         */
        unsigned int init();

        /**
         * @brief Sets the given seed for the random number generator
         *
         * @param seed seed to use
         * @return seed used
         */
        unsigned int init(unsigned int seed);

        /**
         * @brief Returns a random number of given type on the range [min, max]
         *
         * @tparam T type
         * @param min minimal value
         * @param max maximal value
         * @return random value
         */
        template <typename T>
        integer random(T min, T max);

        /**
         * @brief Returns a random number of given type on the range [min, max]
         *
         * @tparam T type
         * @param min minimal value
         * @param max maximal value
         * @return random value
         */
        template <typename T>
        decimal random(T min, T max);

    }
}

#include "rng.inl"
