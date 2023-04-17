#include "rng.h"

#include <random>

static std::mt19937 rng; // thread_local ?

unsigned int RNG::seed(unsigned int seed)
{
    if (seed == NULL)
        seed = std::random_device{}();
    rng.seed(seed);
    return seed;
}

int RNG::number(int min, int max)
{
    std::uniform_int_distribution<int> dist(min, max);
    return dist(rng);
}

float RNG::decimal(float min, float max)
{
    std::uniform_real_distribution<float> dist(min, max);
    return dist(rng);
}
