#include "rng.hpp"

#include <random>

namespace CellularAutomata
{
    namespace random
    {
        static std::default_random_engine _gen(std::random_device{}());

        void seed(unsigned int seed)
        {
            _gen.seed(seed);
        }

        template <typename T>
        integer random(T min, T max)
        {
            std::uniform_int_distribution<T> dis(min, max);
            return dis(_gen);
        }

        template <typename T>
        decimal random(T min, T max)
        {
            std::uniform_real_distribution<T> dis(min, max);
            return dis(_gen);
        }
    }
}
