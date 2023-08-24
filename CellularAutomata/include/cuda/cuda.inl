#pragma once

#include "cuda.cuh"
#include "../core/common.hpp"
#include "../core/exceptions.hpp"

#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define cudaCheck(a) {if (a != cudaSuccess) {throw CellularAutomata::exception::CudaRuntimeError(cudaGetErrorString(cudaGetLastError()));}}

#define TILE_SIZE 32

template<typename T, typename Activation>
__global__ void convKernel(
    const T* __restrict__ input,
    const T* __restrict__ kernel,
    T* output,
    Activation fn,
    const int iHeight,
    const int iWidth,
    const int kSize,
    const bool recursive)
{
    extern __shared__ T shared[];

#pragma region Variables

    // extra space needed because convolution goes beyond borders
    const int halo = kSize / 2;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Global position (aka output element which the thread handles)
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    const int threadsPerBlock = blockDim.y * blockDim.x;

    const int sharedHeight = blockDim.y + kSize - 1;
    const int sharedWidth = blockDim.x + kSize - 1;
    const int memSize = sharedHeight * sharedWidth;


    // rect in which the convolution should be done
    const uint4 rect
    {
        blockIdx.x * blockDim.x,                             // x
        blockIdx.y * blockDim.y,                             // y
        min(iWidth - (blockIdx.x * blockDim.x), blockDim.x), // z
        min(iHeight - (blockIdx.y * blockDim.y), blockDim.y) // w
    };

#pragma endregion Variables

    // TODO: (non) recursive loading 

#pragma region Loading

    // Because of halo / padding, each thread has to load multiple elements
    for (int i = tid; i < memSize; i += threadsPerBlock)
    {
        int y = i / sharedWidth;
        int x = i % sharedWidth;

        if (y < (rect.w + kSize - 1) && x < (rect.z + kSize - 1))
        {
            if (blockIdx.y == 0 && y < halo)
            {
                y += iHeight;
            }
            else if (blockIdx.y == gridDim.y - 1 && y >= TILE_SIZE + halo)
            {
                y -= iHeight;
            }

            if (blockIdx.x == 0 && x < halo)
            {
                x += iWidth;
            }
            else if (blockIdx.x == gridDim.x - 1 && x >= TILE_SIZE + halo)
            {
                x -= iWidth;
            }

            shared[i] = input[(rect.y + y - halo) * iWidth + (rect.x + x - halo)];
        }
    }

#pragma endregion Loading

    __syncthreads();

#pragma region Matrix mul

    // Ignore empty / padded elements (input size is not perfect -> some tiles are not full)
    if (threadIdx.x < rect.z && threadIdx.y < rect.w)
    {
        T temp = 0;

        // Iterate over kernel
        for (int y = 0; y < kSize; y++)
        {
            for (int x = 0; x < kSize; x++)
            {
                int _shared = (threadIdx.y + y) * sharedWidth + (threadIdx.x + x);

                temp += shared[_shared] * kernel[y * kSize + x];
            }
        }

        output[row * iWidth + col] = fn(temp);
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
            const int h, const int w, const int s, const int r)
        {
            dim3 grid(((w - 1) / TILE_SIZE) + 1, ((h - 1) / TILE_SIZE) + 1);
            dim3 block(TILE_SIZE, TILE_SIZE);
            size_t memSize = (TILE_SIZE + s - 1) * (TILE_SIZE + s - 1) * sizeof(T);

            // printDim3(grid);
            // printDim3(block);
            // std::cout << memSize << std::endl;

            convKernel<<<grid, block, memSize>>> (
                input, kernel, output, fn,
                h, w, s, r);

            cudaCheck(cudaDeviceSynchronize());
        }
    }
}
