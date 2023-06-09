#include "backend_cuda.h"

#include "common.h"
#include "activations.h"

#include <iostream>

#include <cstdio>

#define TILE_SIZE 10

#define cudaCheck(a) {if (a != cudaSuccess) {printf("CUDA error (%i): %s\n", __LINE__, cudaGetErrorString(cudaGetLastError()));}}

// CUDA KERNEL

__device__ void cudaPrint2D(float *array, int height, int width, int blockX)
{
    if (threadIdx.y + threadIdx.x == 0 && blockIdx.x == blockX)
    {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                printf("%4.1f ", array[y * width + x]);
            }
            printf("\n");
        }
    }
    __syncthreads();
}

template <typename T>
__device__ activationFunc<T> activFunc[] = {Activations::_normal, Activations::_life};

template<typename T>
__global__ void convKernel(
    const T *__restrict__ input,
    const T *__restrict__ kernel,
    T *output,
    const int op,
    const int iHeight,
    const int iWidth,
    const int kSize)
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

        // T f = activFunc<T>[op](temp);

        // output[row * iWidth + col] = activFunc<T>[op](temp);
        output[row * iWidth + col] = activFunc<T>[op](temp);
    }

    #pragma endregion

}

template __global__ void convKernel(const float *__restrict__, const float *__restrict__, float *, const int, const int, const int, const int);

// CUDA KERNEL

void allocateCUDA(void **array, size_t bytes)
{
    cudaCheck(cudaMalloc(array, bytes));
}

void allocateHost(void **array, size_t bytes)
{
    cudaCheck(cudaMallocHost(array, bytes));
}

void freeCUDA(void *array)
{
    cudaCheck(cudaFree(array));
}

void freeHost(void *array)
{
    cudaCheck(cudaFreeHost(array));
}

void copyCUDA(void *src, Device from, void *dst, Device to, size_t bytes)
{
    cudaMemcpyKind kind;

    if (from == Device::CUDA)
    {
        if (from == Device::CUDA)
        {
            kind = cudaMemcpyKind::cudaMemcpyDeviceToDevice;
        }
        else
        {
            kind = cudaMemcpyKind::cudaMemcpyDeviceToHost;
        }
    }
    else
    {
        if (from == Device::CUDA)
        {
            kind = cudaMemcpyKind::cudaMemcpyHostToDevice;
        }
        else
        {
            kind = cudaMemcpyKind::cudaMemcpyHostToHost;
        }
    }

    cudaCheck(cudaMemcpy(dst, src, bytes, kind));
}

void printDim3(dim3 d)
{
    std::cout << d.x << ", " << d.y << ", " << d.z << std::endl;
}

template <typename T>
void epochCUDA(
    T *input,
    T *kernel,
    T *output,
    const int op,
    const int h,
    const int w,
    const int s)
{
    dim3 grid(((w - 1) / TILE_SIZE) + 1, ((h - 1) / TILE_SIZE) + 1);
    dim3 block(TILE_SIZE, TILE_SIZE);
    size_t memSize = (TILE_SIZE + s - 1) * (TILE_SIZE + s - 1) * sizeof(T);

    // printDim3(grid);
    // printDim3(block);
    // std::cout << memSize << std::endl;

    convKernel<<<grid, block, memSize>>>(
        input, kernel, output, op,
        h, w, s);

    cudaCheck(cudaDeviceSynchronize());
}

template void epochCUDA(float *, float *, float *, const int, const int, const int, const int);
