#include "rng.h"

#include <random>
#include <type_traits>

static std::mt19937 rng; // thread_local ?

unsigned int RNG::seed(unsigned int seed)
{
    if (seed == NULL)
        seed = std::random_device{}();
    rng.seed(seed);
    return seed;
}

template <typename T>
integer RNG::random(T min, T max)
{
    std::uniform_int_distribution<T> dist(min, max);
    return dist(rng);
}

template <typename T>
decimal RNG::random(T min, T max)
{
    std::uniform_real_distribution<T> dist(min, max);
    return dist(rng);
}
