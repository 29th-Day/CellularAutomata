#pragma once

#include "../core/common.hpp"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

/**
 * @defgroup custom_activations Custom Activations
 *
 * Shorthand for creating a class and overloading the 'operator()'-function.
 * @note Functors are needed because CUDA function pointers are not (easily) accessible from host code.
 *
 * @{
 */

 /**
 * @brief Macro for defining custom activations
 * @param name Functor name
 * @param body Function body (input as parameter 'x')
 *
 */
#define ACTIVATION(name, body)                      \
    template <typename T>                           \
    class name                                      \
    {                                               \
    public:                                         \
        cudaFn T operator()(const T x) const body   \
    };

 /** @} */ // End of custom_activations group

namespace CellularAutomata
{
    /**
     * @brief Commonly used activation functions
     */
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
            /**
             * @brief Functor call
             * @param x input
             * @return transformed x
             */
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
            /**
             * @brief Functor call
             * @param x input
             * @return transformed x
             */
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
            /**
             * @brief Functor call
             * @param x input
             * @return transformed x
             */
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
            /**
             * @brief Functor call
             * @param x input
             * @return transformed x
             */
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
            /**
             * @brief Functor call
             * @param x input
             * @return transformed x
             */
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
            /**
             * @brief Functor call
             * @param x input
             * @return transformed x
             */
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
            /**
             * @brief Functor call
             * @param x input
             * @return transformed x
             */
            cudaFn T operator()(const T x) const;
        };
    }
}

#include "activations.inl"
