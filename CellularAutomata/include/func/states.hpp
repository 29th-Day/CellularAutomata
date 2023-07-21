#pragma once

#include "../core/common.hpp"

namespace CellularAutomata
{
    namespace States
    {
        /**
         * @brief Fills the kernel with values between [0, 1]
         *
         * @tparam T type
         * @param array current state array
         * @param h state height
         * @param w state width
         */
        template <typename T>
        void normal(T* array, const int h, const int w);

        /**
         * @brief Fills the kernel with either 0 or 1
         *
         * @tparam T type
         * @param array current state array
         * @param h state height
         * @param w state width
         */
        template <typename T>
        void binary(T* array, const int h, const int w);

        /**
         * @brief Some predefined objects of Conway's Game of Life.
         */
        namespace Objects
        {
            /**
             * @brief Copy an (small) array into another (large) array at a given position
             * @attention No checks are done if the size is valid at the given position
             * @note The copy is top-left aligned
             *
             * @tparam T type
             * @param src small source array
             * @param src_h source height
             * @param src_w source width
             * @param dst large destination array
             * @param dst_w destination width
             * @param y Y insert position
             * @param x X insert position
             */
            template <typename T>
            inline void copyInto(T* src, const int src_h, const int src_w, T* dst, const int dst_w, const int y, const int x);

            /**
             * @brief Copies a glider (3x3) into given position (https://conwaylife.com/wiki/Glider)
             * @attention State MUST be on CPU. No checks are done if the size is valid at the given position
             * @note The copy is top-left aligned
             *
             * @tparam T type
             * @param state state on CPU
             * @param x X-position
             * @param y Y-position
             */
            template <typename T>
            void Glider(State<T>* state, const int x, const int y);

            /**
             * @brief Copies a spaceship (4x5) into given position (https://conwaylife.com/wiki/Spaceship)
             * @attention State MUST be on CPU. No checks are done if the size is valid at the given position
             * @note The copy is top-left aligned
             *
             * @tparam T type
             * @param state state on CPU
             * @param x X-position
             * @param y Y-position
             */
            template <typename T>
            void Spaceship(State<T>* state, const int x, const int y);

            /**
             * @brief Copies a bipole (4x4) into given position (https://conwaylife.com/wiki/Bipole)
             * @attention State MUST be on CPU. No checks are done if the size is valid at the given position
             * @note The copy is top-left aligned
             *
             * @tparam T type
             * @param state state on CPU
             * @param x X-position
             * @param y Y-position
             */
            template <typename T>
            void Bipole1(State<T>* state, const int x, const int y);

            /**
             * @brief Copies a (alternative) bipole (5x5) into given position (https://conwaylife.com/wiki/Bipole)
             * @attention State MUST be on CPU. No checks are done if the size is valid at the given position
             * @note The copy is top-left aligned
             *
             * @tparam T type
             * @param state state on CPU
             * @param x X-position
             * @param y Y-position
             */
            template <typename T>
            void Bipole2(State<T>* state, const int x, const int y);

            /**
             * @brief Copies a R-pentomino (3x3) into given position (https://conwaylife.com/wiki/R-pentomino)
             * @attention State MUST be on CPU. No checks are done if the size is valid at the given position
             * @note The copy is top-left aligned
             *
             * @tparam T type
             * @param state state on CPU
             * @param x X-position
             * @param y Y-position
             */
            template <typename T>
            void r_Pentomino(State<T>* state, const int x, const int y);
        }
    }

}

#include "states.inl"
