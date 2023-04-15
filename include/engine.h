/**
 * @file engine.h
 * @author 29th-Day (https://github.com/29th-Day)
 * @version 0.1
 * @date 2023-04-15
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include "common.h"
#include "states.h"
#include "kernels.h"
#include "activations.h"

namespace Engine
{
    /**
     * @brief Initializes RNG
     * @details Initializes RNG and sets seed for reproducible outcomes.
     * @param seed RNG seed
     */
    void InitRandom(unsigned int seed);

    /**
     * @brief Initializes a state
     * @details description
     * @param state reference to a state
     * @param height height of the world
     * @param width width of the world
     * @param f function for initialising the world
     */
    void InitState(State *state, int height, int width, state_func f);

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
     * @param f function for initialising the kernel
     */
    void InitKernel(Kernel *kernel, kernel_func f);

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
     */
    void Epoch(State *state, Kernel *kernel, activation_func f);
}