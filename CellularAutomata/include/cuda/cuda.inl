#pragma once

#include "cuda.cuh"
#include "../core/common.hpp"
#include "../core/exceptions.hpp"

#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define cudaCheck(a) do {if ((a) != cudaSuccess) throw CellularAutomata::exception::CudaRuntime(); } while(false)

#define TILE_SIZE 32

/**
 * @brief Positive modulo - a % n
 * @param a dividend
 * @param n divisor
 * @return unsigned int
 *
 * More infomation on [modulo](https://en.wikipedia.org/wiki/Modulo)
 */
__device__ __forceinline__ const unsigned int modP(const int a, const int n)
{
    int r = a % n;
    if (r < 0)
        r += n;
    return r;
}

template<typename T>
__device__ void loadShared_recursive(const T* __restrict__ input, T* shared, const uint2 inputShape, const uint2 sharedShape, const unsigned int kernelSize, const uint4 valid)
{
    const int halo = kernelSize / 2;                        // extra space needed because convolution goes beyond borders
    const int tid = threadIdx.y * blockDim.x + threadIdx.x; // local thread ID
    const int memSize = sharedShape.y * sharedShape.x;      // shared memory size
    const int threadsPerBlock = blockDim.y * blockDim.x;    // total number of threads per block (should be TILE_SIZE * TILE_SIZE)

    // Because of halo / padding, each thread has to load multiple elements
    for (int i = tid; i < memSize; i += threadsPerBlock)
    {
        // convert i back to local x,y coordinates
        int y = i / sharedShape.x;
        int x = i % sharedShape.x;

        // check if x,y are inside the valid area
        if (y < (valid.w + kernelSize - 1) && x < (valid.z + kernelSize - 1))
        {
            // 1. convert x,y to global x,y
            // 2. loop x,y if outside of input array range
            y = modP(y + valid.y - halo, inputShape.y);
            x = modP(x + valid.x - halo, inputShape.x);

            // copy
            shared[i] = input[y * inputShape.x + x];
        }
    }
}

template<typename T>
__device__ void loadShared_simple(const T* __restrict__ input, T* shared, const int inputWidth, const int sharedWidth, const unsigned int kernelSize, const uint4 valid)
{
    const int halo = kernelSize / 2;    // extra space needed because convolution goes beyond borders

    if (threadIdx.y < valid.w && threadIdx.x < valid.z)
    {
        // each thread loads their element into shared offset by halo
        shared[(threadIdx.y + halo) * sharedWidth + (threadIdx.x + halo)] = input[threadIdx.y * inputWidth + threadIdx.x];
    }
}

template<typename T, typename Activation>
__global__ void convKernel(
    const T* __restrict__ input,
    const T* __restrict__ kernel,
    T* output,
    Activation fn,
    const uint2 inputShape,
    const unsigned int kernelSize,
    const bool recursive)
{
    extern __shared__ T shared[];

#pragma region Variables

    // Global position (aka output element which the thread handles): x = col, y = row
    const uint2 pos{ blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y };

    // shared memory shape
    const uint2 sharedShape{ blockDim.x + kernelSize - 1, blockDim.y + kernelSize - 1 };

    // rect in which the convolution should be done
    const uint4 valid
    {
        blockIdx.x * blockDim.x,                                    // x
        blockIdx.y * blockDim.y,                                    // y
        min(inputShape.x - (blockIdx.x * blockDim.x), blockDim.x),  // z
        min(inputShape.y - (blockIdx.y * blockDim.y), blockDim.y)   // w
    };

#pragma endregion Variables


#pragma region Loading

    if (recursive)
        loadShared_recursive(input, shared, inputShape, sharedShape, kernelSize, valid);
    else
        loadShared_simple(input, shared, inputShape.x, sharedShape.x, kernelSize, valid);

#pragma endregion Loading

    __syncthreads();

#pragma region Matrix mul

    // Ignore empty / padded elements (input size is not perfect -> some tiles are not full)
    if (threadIdx.x < valid.z && threadIdx.y < valid.w)
    {
        T temp = 0;

        // Iterate over kernel
        for (int y = 0; y < kernelSize; y++)
        {
            for (int x = 0; x < kernelSize; x++)
            {
                int _shared = (threadIdx.y + y) * sharedShape.x + (threadIdx.x + x);

                temp += shared[_shared] * kernel[y * kernelSize + x];
            }
        }

        output[pos.y * inputShape.x + pos.x] = fn(temp);
    }

#pragma endregion

}

namespace CellularAutomata
{
    namespace cuda
    {
        void allocateCUDA(void** arrayPtr, size_t bytes)
        {
            cudaCheck(cudaMalloc(arrayPtr, bytes));
        }

        void allocateHost(void** arrayPtr, size_t bytes)
        {
            cudaCheck(cudaMallocHost(arrayPtr, bytes));
        }

        void freeCUDA(void* array)
        {
            cudaCheck(cudaFree(array));
        }

        void freeHost(void* array)
        {
            cudaCheck(cudaFreeHost(array));
        }

        void copyCUDA(void* src, Device from, void* dst, Device to, size_t bytes)
        {
            cudaCheck(cudaMemcpy(dst, src, bytes, cudaMemcpyKind::cudaMemcpyDefault));
        }

        template <typename T, typename Activation>
        void epoch(
            T* input, T* kernel, T* output, Activation fn,
            const unsigned int h, const unsigned int w, const unsigned int s, const bool r)
        {
            dim3 grid(((w - 1) / TILE_SIZE) + 1, ((h - 1) / TILE_SIZE) + 1);
            dim3 block(TILE_SIZE, TILE_SIZE);
            size_t memSize = (TILE_SIZE + s - 1) * (TILE_SIZE + s - 1) * sizeof(T);

            // printDim3(grid);
            // printDim3(block);
            // std::cout << memSize << std::endl;

            convKernel<<<grid, block, memSize>>> (
                input, kernel, output, fn,
                uint2{ h, w }, s, r);

            cudaError_t e = cudaDeviceSynchronize();

            if (e != cudaSuccess)
                printf("%s: %s\n", cudaGetErrorName(e), cudaGetErrorString(e));

            // throw CellularAutomata::exception::CudaRuntime(cudaGetErrorString(e));

            // cudaCheck(cudaDeviceSynchronize());
        }
    }
}

#undef TILE_SIZE
#undef cudaCheck
