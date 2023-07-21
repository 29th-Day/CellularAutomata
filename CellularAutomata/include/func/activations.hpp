#pragma once

#include "../core/common.hpp"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

/**
 * @brief Macro for defining custom activations. Shorthand for creating a class and overloading the 'operator()'-function.
 * @param name Functor name
 * @param body Function body (input as parameter 'x')
 * @note Functors are needed because CUDA function pointers are not (easily) accessible from host code
 */
#define ACTIVATION(name, body)                      \
    template <typename T>                           \
    class name                                      \
    {                                               \
    public:                                         \
        cudaFn T operator()(const T x) const body   \
    };                                              \

namespace CellularAutomata
{
    namespace Activations
    {
        /**
         * @brief Hard clamp between [0,1]
         *
         * @tparam T type
         */
        template<typename T>
        class normal
        {
        public:
            cudaFn T operator()(const T x) const;
        };

        /**
         * @brief Rules for Conway's Game of Life
         *
         * @tparam T type
         */
        template<typename T>
        class life
        {
        public:
            cudaFn T operator()(const T x) const;
        };

        /**
         * @brief Smooth clamp between [0, 1]
         *
         * @tparam T type
         */
        template<typename T>
        class sigmoid
        {
        public:
            cudaFn T operator()(const T x) const;
        };

        /**
         * @brief Smooth clamp between [-1, 1]
         *
         * @tparam T type
         */
        template<typename T>
        class tanh
        {
        public:
            cudaFn T operator()(const T x) const;
        };

        /**
         * @brief Periodic clamp between [-1, 1]
         *
         * @tparam T  type
         */
        template<typename T>
        class sin
        {
        public:
            cudaFn T operator()(const T x) const;
        };

        /**
         * @brief Periodic clamp between [-1, 1]
         *
         * @tparam T  type
         */
        template<typename T>
        class cos
        {
        public:
            cudaFn T operator()(const T x) const;
        };

        /**
         * @brief Periodic clamp between (-inf, inf)
         *
         * @tparam T  type
         */
        template<typename T>
        class tan
        {
        public:
            cudaFn T operator()(const T x) const;
        };
    }
}

#include "activations.inl"
