#include "activations.hpp"
#include "../core/common.hpp"
#include <cmath>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#define EULER_NUMBER_F 2.71828182846f

namespace CellularAutomata
{
    namespace Activations
    {
        template<typename T>
        cudaFn T normal<T>::operator()(const T x) const
        {
            return static_cast<T>((x < 0) ? 0 : (x > 1 ? 1 : x));
        }

        template<typename T>
        cudaFn T life<T>::operator()(const T x) const
        {
            unsigned char u = static_cast<unsigned char>(x);

            unsigned char neighbours = u & 0xF; // low  nibble
            bool alive = u & 0xF0;              // high nibble

            switch (neighbours)
            {
            case 2:
                // staying alive
                return static_cast<T>((alive) ? 1 : 0);
            case 3:
                // birth
                return static_cast<T>(1);
            default:
                // under- / overpopulation
                return static_cast<T>(0);
            }
        }

        template<typename T>
        cudaFn T sigmoid<T>::operator()(const T x) const
        {
            return (1 / (1 + std::pow(EULER_NUMBER_F, -x)));
        }

        template<typename T>
        cudaFn T tanh<T>::operator()(const T x) const
        {
            return std::tanh(x);
        }

        template<typename T>
        cudaFn T sin<T>::operator()(const T x) const
        {
            return std::sin(x);
        }

        template<typename T>
        cudaFn T cos<T>::operator()(const T x) const
        {
            return std::cos(x);
        }

        template<typename T>
        cudaFn T tan<T>::operator()(const T x) const
        {
            return std::tan(x);
        }
    }
}
