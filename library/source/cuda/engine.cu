#ifdef CUDA

#include "CellularAutomata.h"

#include <stdlib.h>

#include <cuda_runtime.h>

#define TILE_SIZE 32

#define GPU 0
#define CPU cudaCpuDeviceId

// CUDA Kernels

__global__ void convolution(const float *__restrict__ input, const float *__restrict__ kernel, float *output, activation_func fn, const int iHeight, const int iWidth, const int kSize)
{
    extern __shared__ float shared[];

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

    // cudaPrint4(rect);


    // Because of halo / padding, each thread has to load multiple elements
    for (int i = tid; i < memSize; i += threadsPerBlock)
    {
        int y = i / sharedWidth;
        int x = i % sharedWidth;

        if (y < (rect.w + kSize - 1) && x < (rect.z + kSize - 1))
        {
            if (y < halo)
            {
                y += iHeight;
            }
            else if (rect.y + y >= iHeight + halo)
            {
                y -= iHeight;
            }
            
            if (x < halo)
            {
                x += iWidth;
            }
            else if (rect.x + x >= iWidth + halo)
            {
                x -= iWidth;
            }

            shared[i] = input[(rect.y + y - halo) * iWidth + (rect.x + x - halo)];
        }
    }

    __syncthreads();

    // Compute kernel

    // Ignore empty / padded elements (input size is not perfect -> some tiles are not full)
    if (threadIdx.x < rect.z && threadIdx.y < rect.w)
    {
        float temp = 0.0f;

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
}

// Engine

void CellularAutomata::InitState(State *state, unsigned int height, unsigned int width, state_func f)
{
    state->height = height;
    state->width = width;
    cudaMallocManaged(&state->current, height * width * sizeof(float));
    cudaMallocManaged(&state->next, height * width * sizeof(float));
    if (f != NULL)
        f(state);
}

void CellularAutomata::DestroyState(State *state)
{
    cudaFree(state->current);
    state->current = nullptr;
    cudaFree(state->next);
    state->next = nullptr;
    state->height = 0;
    state->width = 0;
}

void CellularAutomata::InitKernel(Kernel *kernel, unsigned int kernelSize, kernel_func f)
{
    cudaMallocManaged(&kernel->kernel, kernelSize * kernelSize * sizeof(float));
    kernel->size = kernelSize;

    if (f != NULL)
        f(kernel);
}

void CellularAutomata::DestroyKernel(Kernel *kernel)
{
    cudaFree(kernel->kernel);
    kernel->kernel = nullptr;
    kernel->size = 0;
}

void CellularAutomata::Epoch(State *state, Kernel *kernel, activation_func f, bool recursive)
{
    size_t bytes_matrix = state->height * state->width * sizeof(float);
    size_t bytes_kernel = kernel->size * kernel->size * sizeof(float);

    cudaMemPrefetchAsync(state->current, bytes_matrix, GPU);
    cudaMemPrefetchAsync(state->next, bytes_matrix, GPU);
    cudaMemPrefetchAsync(kernel->kernel, bytes_kernel, GPU);

    dim3 grid(((state->width - 1) / TILE_SIZE) + 1, ((state->height - 1) / TILE_SIZE) + 1);
    dim3 block(TILE_SIZE, TILE_SIZE);
    size_t memSize = (TILE_SIZE + kernel->size - 1) * (TILE_SIZE + kernel->size - 1) * sizeof(float);

    convolution<<<grid, block, memSize>>>(state->current, kernel->kernel, state->next, f, state->height, state->width, kernel->size);

    cudaDeviceSynchronize();
}

#endif
