#include "common.hpp"
#include "exceptions.hpp"

#ifdef __CUDACC__
#include "../cuda/cuda.cuh"
#endif

#include <algorithm>
#include <iostream>

#define stateSize height * width * sizeof(T)
#define kernelSize size * size * sizeof(T)

namespace
{
    template <typename T>
    inline void _print2D(T* array, const unsigned int h, const unsigned int w)
    {
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                std::cout << array[y * w + x] << " ";
            }
            std::cout << std::endl;
        }
    }
}

namespace CellularAutomata
{
    template <typename T>
    State<T>::State(const unsigned int height, const unsigned int width, stateFunc<T> fn, Device device) : height(height), width(width), device(device)
    {
    #ifdef __CUDACC__
        if (sizeof(T) != 4)
            throw exceptions::TypeNotSupported("Only 32bit types are supported with CUDA");

        initCUDA(fn);
    #else
        if (device == Device::CUDA)
            throw exceptions::DeviceNotAvailable("CUDA");

        initCPU(fn);
    #endif
    }

    template <typename T>
    void State<T>::initCPU(stateFunc<T> fn)
    {
        curr = new T[height * width]();
        next = new T[height * width]();

        if (fn != nullptr)
            fn(curr, height, width);
    }

    template <typename T>
    void State<T>::initCUDA(stateFunc<T> fn)
    {
        size_t bytes = stateSize;

        if (device == Device::CUDA)
        {
            // allocate GPU memory

            CellularAutomata::cuda::allocateCUDA(reinterpret_cast<void**>(&curr), bytes);
            CellularAutomata::cuda::allocateCUDA(reinterpret_cast<void**>(&next), bytes);

            if (fn != nullptr)
            {
                T* temp = new T[height * width]();

                fn(temp, height, width);
                CellularAutomata::cuda::copyCUDA(temp, Device::CPU, curr, Device::CUDA, bytes);

                delete[] temp;
            }
        }
        else
        {
            // allocate (pinned) CPU memory (for faster transfer)
            // just always use pinned memory when CUDA is enabled (maybe use condition?)

            CellularAutomata::cuda::allocateHost(reinterpret_cast<void**>(&curr), bytes);
            CellularAutomata::cuda::allocateHost(reinterpret_cast<void**>(&next), bytes);

            if (fn != nullptr)
                fn(curr, height, width);
        }
    }

    template <typename T>
    State<T>::~State()
    {
    #ifdef __CUDACC__
        freeCUDA();
    #else
        freeCPU();
    #endif
    }

    template <typename T>
    void State<T>::freeCPU()
    {
        delete[] curr;
        delete[] next;
    }

    template <typename T>
    void State<T>::freeCUDA()
    {
        if (device == Device::CUDA)
        {
            CellularAutomata::cuda::freeCUDA(curr);
            CellularAutomata::cuda::freeCUDA(next);
        }
        else
        {
            CellularAutomata::cuda::freeHost(curr);
            CellularAutomata::cuda::freeHost(next);
        }
    }

    template <typename T>
    void State<T>::copyTo(State<T>* to)
    {
        if (height != to->height && width != to->width)
            throw exceptions::ShapesUnequal();

    #ifdef __CUDACC__
        if (device == Device::CUDA || to->device == Device::CUDA)
        {
            size_t bytes = stateSize;
            CellularAutomata::cuda::copyCUDA(curr, device, to->curr, to->device, bytes);
        }
    #endif

        std::copy(curr, curr + height * width, to->curr);
    }

    template <typename T>
    void State<T>::swap()
    {
        auto tmp = curr;
        curr = next;
        next = tmp;
    }

    template <typename T>
    void State<T>::print()
    {
    #ifdef __CUDACC__
        if (device == Device::CUDA)
        {
            // TODO: pinned memory is slower to allocate but faster to transfer
            size_t bytes = stateSize;
            T* temp = new T[height * width]();
            CellularAutomata::cuda::copyCUDA(curr, device, temp, Device::CPU, bytes);

            _print2D(temp, height, width);

            delete[] temp;

            return;
        }
    #endif

        _print2D(curr, height, width);
    }


    template <typename T>
    Kernel<T>::Kernel(const unsigned int size, kernelFunc<T> fn, Device device) : size(size), device(device)
    {
        if (device == Device::CUDA)
        {
        #ifdef __CUDACC__
            initCUDA(fn);
        #else
            throw exceptions::DeviceNotAvailable("CUDA");
        #endif
        }
        else
        {
            initCPU(fn);
        }
    }

    template <typename T>
    void Kernel<T>::initCPU(kernelFunc<T> fn)
    {
        kernel = new T[size * size]();

        if (fn != nullptr)
            fn(kernel, size);
    }

    template <typename T>
    void Kernel<T>::initCUDA(kernelFunc<T> fn)
    {
        size_t bytes = kernelSize;

        CellularAutomata::cuda::allocateCUDA(reinterpret_cast<void**>(&kernel), bytes);

        if (fn != nullptr)
        {
            T* temp = new T[size * size]();

            fn(temp, size);
            CellularAutomata::cuda::copyCUDA(temp, Device::CPU, kernel, Device::CUDA, bytes);

            delete[] temp;
        }
    }

    template <typename T>
    Kernel<T>::~Kernel()
    {
    #ifdef __CUDACC__
        if (device == Device::CUDA)
        {
            freeCUDA();
            return;
        }
    #endif

        freeCPU();
    }

    template <typename T>
    void Kernel<T>::freeCPU()
    {
        delete[] kernel;
    }

    template <typename T>
    void Kernel<T>::freeCUDA()
    {
        CellularAutomata::cuda::freeCUDA(kernel);
    }

    template <typename T>
    void Kernel<T>::copyTo(Kernel<T>* to)
    {
        if (size != to->size)
            throw exceptions::ShapesUnequal();

    #ifdef __CUDACC__
        if (device == Device::CUDA)
        {
            size_t bytes = kernelSize;
            CellularAutomata::cuda::copyCUDA(kernel, device, to->kernel, to->device, bytes);
            return;
        }
    #endif

        std::copy(kernel, kernel + size * size, to->kernel);
    }

    template <typename T>
    void Kernel<T>::print()
    {
    #ifdef __CUDACC__
        if (device == Device::CUDA)
        {
            size_t bytes = kernelSize;
            T* temp = new T[size * size]();
            CellularAutomata::cuda::copyCUDA(kernel, device, temp, Device::CPU, bytes);

            _print2D(temp, size, size);

            delete[] temp;

            return;
        }
    #endif

        _print2D(kernel, size, size);
    }
}

#undef stateSize
#undef kernelSize
