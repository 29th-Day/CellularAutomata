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
using stateFunc = void (*)(T*, const unsigned int, const unsigned int);

template <typename T>
/**
 * @brief Universal function pointer alias for function which populate a kernel.
 * @tparam T type
 * @param array kernel array
 * @param size size of kernel (always square)
 */
using kernelFunc = void (*)(T*, const unsigned int);

/**
 * @brief Main namespace
 */
namespace CellularAutomata
{
    /**
     * @brief Computational device
     * @note A object allocated on a specific device can only interact with objects on the same device.
     */
    enum Device
    {
        /**
         * @brief Allocation & computation on CPU/RAM
         */
        CPU,
        /**
         * @brief Allocation & computation on NVIDIA GPU via CUDA
         */
        CUDA,
    };

    /**
     * @brief 2D buffered matrices
     * @tparam T type
     */
    template <typename T>
    class State
    {
    public:
        /**
         * @brief Current 2D Matrix buffer
         */
        T* curr;

        /**
         * @brief Next 2D Matrix buffer
         */
        T* next;

        /**
         * @brief Height of 2D Matrix
         */
        const unsigned int height;

        /**
         * @brief Width of 2D Matrix
         */
        const unsigned int width;

        /**
         * @brief Computational device
         */
        const Device device;

        /**
         * @brief Initalizes a state
         *
         * @param height height of the grid world
         * @param width width of the grid world
         * @param fn function to fill the current state buffer. May be 'nullptr'
         * @param device computational device
         * @throw DeviceNotAvailable
         */
        State(const unsigned int height, const unsigned int width, stateFunc<T> fn, Device device);

        /**
         * @brief Frees dynamically allocated memory
         */
        ~State();

        /**
         * @brief Copy the own current state buffer to another current state buffer across devices
         * @param to other state
         * @throw ShapesUnequal
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

    /**
     * @brief 2D convolution kernel
     *
     * @tparam T type
     */
    template <typename T>
    class Kernel
    {
    public:
        /**
         * @brief 2D Matrix
         */
        T* kernel;
        /**
         * @brief Side length of matrix
         */
        const unsigned int size;
        /**
         * @brief Computational device
         */
        const Device device;

        /**
         * @brief Initalizes a kernel
         *
         * @param size side length of kernel (always square)
         * @param fn function to fill the kernel. May be 'nullptr'
         * @param device computational device
         * @throw DeviceNotAvailable
         */
        Kernel(const unsigned int size, kernelFunc<T> fn, Device device);

        /**
         * @brief Frees dynamically allocated memory
         */
        ~Kernel();

        /**
         * @brief Copy the own kernel array to another kernel array across devices
         * @param to other kernel
         * @throw ShapesUnequal
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
