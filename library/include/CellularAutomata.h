/**
 * @file CellularAutomata.h
 * @author 29th-Day (https://github.com/29th-Day)
 *
 * @copyright Copyright (c) 2023
 *
 * This software is licensed under the MIT License
 */

#pragma once

// include all to be accessible from outside
#include "common.h"
#include "rng.h"
#include "states.h"
#include "kernels.h"
#include "activations.h"

namespace CellularAutomata
{
    /**
     * @brief Initializes RNG
     * @details Seeds RNG for reproducible outcomes. Seed is randomized if NULL.
     * @param seed RNG seed
     * @return used seed
     */
    inline unsigned int InitRandom(unsigned int seed) { return RNG::seed(seed); }

    /**
     * @brief Initializes a state
     * @details description
     * @param state reference to a state
     * @param height height of the world
     * @param width width of the world
     * @param f function for initialising the world. May be NULL
     */
    void InitState(State *state, unsigned int height, unsigned int width, state_func f);

    /**
     * @brief Destroy state
     * @details Free the dynamicly allocated fields of a state.
     * @param state reference to a state
     */
    void DestroyState(State *state);

    /**
     * @brief Initializes a kernel
     * @details description
     * @param kernel reference to a kernel
     * @param kernelSize length of kernel side
     * @param f function for initialising the kernel
     */
    void InitKernel(Kernel *kernel, unsigned int kernelSize, kernel_func f);

    /**
     * @brief Destroy kernel
     * @details Free the dynamicly allocated fields of a kernel.
     * @param kernel reference to a kernel
     */
    void DestroyKernel(Kernel *kernel);

    /**
     * @brief Runs a epoch on a state
     * @details Runs a convolution with the state and kernel and applies the activation function.
     * @param state reference to a state
     * @param kernel reference to a kernel
     * @param f activation function after convolution
     * @param recursive wrap the convolution around the borders of the world ("infinite grid")
     */
    void Epoch(State *state, Kernel *kernel, activation_func f, bool recursive);
}
