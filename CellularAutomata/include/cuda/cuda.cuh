#pragma once

#include "../core/common.hpp"

namespace CellularAutomata
{
    namespace cuda
    {
        /**
         * @brief Allocated memory on a CUDA device
         * 
         * @param array pointer to deviceArray
         * @param bytes allocation size in bytes
         */
        void allocateCUDA(void** arrayPtr, size_t bytes);

        /**
         * @brief Allocated memory on the host device (CPU)
         * @note Allocates "page-locked memory" which is slower in allocation but faster in transmission between devices and host. Should be used for memory which is often transfered
         * 
         * @param arrayPtr pointer to hostArray
         * @param bytes allocation size in bytes
         */
        void allocateHost(void** arrayPtr, size_t bytes);

        /**
         * @brief  Frees memory on a CUDA device
         * 
         * @param array deviceArray to free
         */
        void freeCUDA(void* array);

        /**
         * @brief Frees memory on the host device (CPU)
         * @attention Do NOT use on regular allocations
         * @note Frees "page-locked memory" allocated with CUDA functions
         * 
         * @param array hostArray to free
         */
        void freeHost(void* array);

        /**
         * @brief Memcopy between devices and host
         * @attention Size is assumed to be equal
         * @note Host memory can be allocated using regular allocations
         * 
         * @param src input array to copy from
         * @param from computational device where input lives
         * @param dst output array to copy to
         * @param to computational device where output lives
         * @param bytes allocation size in bytes
         */
        void copyCUDA(void* src, Device from, void* dst, Device to, size_t bytes);

        /**
         * @brief Call the CUDA kernel to run a single epoch on a given state using a kernel
         * 
         * @tparam T type
         * @tparam Activation functor 
         * @param input current state array
         * @param kernel kernel for convolution
         * @param output next state array
         * @param fn activation function
         * @param h height of state array
         * @param w width of state array
         * @param s size of kernel
         * @param r grid recursion enabled
         */
        template <typename T, typename Activation>
        void epoch(
            T* input, T* kernel, T* output, Activation fn,
            const int h, const int w, const int s, const int r);
    }
}

#include "cuda.inl"
