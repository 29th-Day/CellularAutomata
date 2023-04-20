#pragma once

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

    /**
     * @brief Generates a random int
     *
     * @param min Minimum value of range
     * @param max Maximum value of range
     * @return random integer
     */
    int number(int min = 0, int max = 1);

    /**
     * @brief Generates a random float
     *
     * @param min Minimum value of range
     * @param max Maximum value of range
     * @return random float
     */
    float decimal(float min, float max);
}
