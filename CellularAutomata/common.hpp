#pragma once

// macro for registering functions to CUDA (if used)
#ifdef __CUDACC__
#define cudaFn __device__ __host__
#else
#define cudaFn
#endif

template <typename T>
/**
 * @brief Universal function pointer alias for function which populate a state.
 * @tparam T type
 * @param array current state array
 * @param height height of array
 * @param width width of array
 */
using stateFunc = void (*)(T*, const int, const int);

template <typename T>
/**
 * @brief Universal function pointer alias for function which populate a kernel.
 * @tparam T type
 * @param array kernel array
 * @param size size of kernel (always square)
 */
using kernelFunc = void (*)(T*, const int);

namespace CellularAutomata
{
    /**
     * @brief Computational device
     */
    enum Device
    {
        CPU,
        CUDA,
    };

    /**
     * @brief Class for handeling double buffered 2D grids.
     *
     *
     * @tparam T type
     */
    template <typename T>
    class State
    {
    public:
        T* curr;
        T* next;
        const int height;
        const int width;
        const Device device;

        /**
         * @brief Initalizes a state
         *
         * @param height height of the grid world
         * @param width width of the grid world
         * @param fn function to fill the current state buffer. May be 'nullptr'
         * @param device computational device
         */
        State(int height, int width, stateFunc<T> fn, Device device);

        /**
         * @brief Frees dynamically allocated memory
         */
        ~State();

        /**
         * @brief Copy the own current state buffer to another current state buffer
         * @attention Height and width are assumed to be equal
         * @note States can be on different devices
         *
         * @param to other state
         */
        void copyTo(State<T>* to);

        /**
         * @brief Swaps the current- and next state buffer
         */
        void swap();

        /**
         * @brief Prints the current state buffer
         */
        void print();

    private:
        void initCPU(stateFunc<T> fn);
        void initCUDA(stateFunc<T> fn);
        void freeCPU();
        void freeCUDA();
    };

    template <typename T>
    class Kernel
    {
    public:
        T* kernel;
        const int size;
        const Device device;

        /**
         * @brief Initalizes a kernel
         *
         * @param size side length of kernel (always square)
         * @param fn function to fill the kernel. May be 'nullptr'
         * @param device computational device
         */
        Kernel(int size, kernelFunc<T> fn, Device device);

        /**
         * @brief Frees dynamically allocated memory
         */
        ~Kernel();

        /**
         * @brief Copy the own kernel array to another kernel array
         * @attention Size is assumed to be equal
         * @note Kernels may be on different devices
         *
         * @param to other kernel
         */
        void copyTo(Kernel<T>* to);

        /**
         * @brief Prints the kernel array
         */
        void print();

    private:
        void initCPU(kernelFunc<T> fn);
        void initCUDA(kernelFunc<T> fn);
        void freeCPU();
        void freeCUDA();
    };

}

#include "common.inl"
