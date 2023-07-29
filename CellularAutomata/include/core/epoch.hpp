#pragma once

namespace CellularAutomata
{
    /**
     * @brief Runs a single epoch on a given state using a kernel
     * @note Device must be equal and is determined by the state
     *
     * @tparam T type
     * @tparam Activation functor
     * @param state state of type T
     * @param kernel kernel of type T
     * @param activation functor of type T
     * @param recursive if world should be looping (flat grid behaves like torus surface)
     */
    template <typename T, typename Activation>
    void Epoch(State<T>* state, Kernel<T>* kernel, Activation activation, bool recursive);
}

#include "epoch.inl"
