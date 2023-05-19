
#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define checkCudaError(a)                                                         \
    cudaError_t err = a;                                                          \
    if (err != cudaSuccess)                                                       \
    {                                                                             \
        printf("CUDA error (Line %d):  %s\n", __LINE__, cudaGetErrorString(err)); \
        exit(-1);                                                                 \
    }

#define cudaPrint(...) if (threadIdx.y + threadIdx.x == 0) printf(__VA_ARGS__);

#define cudaPrintThread(a, b, ...) if (threadIdx.y == a && threadIdx.x == b) printf(__VA_ARGS__);

#define cudaPrint4(i) cudaPrint("x: %i, y: %i, w: %i, h: %i\n", i.x, i.y, i.z, i.w)

void print2D(float *array, int height, int width)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            printf("%3.1f ", array[y * width + x]);
        }
        printf("\n");
    }
    printf("\n");
}

__device__ void cudaPrint2D(float *array, int height, int width)
{
    if (threadIdx.y + threadIdx.x == 0)
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

#define TILE_SIZE 32

#define GPU 0
#define CPU cudaCpuDeviceId

__global__ void convolution(const float *__restrict__ input, const float *__restrict__ kernel, float *output, const int iHeight, const int iWidth, const int kSize)
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

        output[row * iWidth + col] = temp;
    }
}

template <typename T>
void fill(T* array, T value, size_t length)
{
    for (size_t i = 0; i < length; i++)
    {
        array[i] = value;
    }
}

template <typename T>
void fillRow(T* array, T value, int row, int height, int width)
{
    for (int x = 0; x < width; x++)
    {
        array[row * width + x] = value;
    }
}

template <typename T>
void fillCol(T* array, T value, int col, int height, int width)
{
    for (int y = 0; y < height; y++)
    {
        array[y * width + col] = value;
    }
}

#define POS(x,y) y * width + x

/* int main()
{
    int width  = 10;
    int height = 5;

    int size = 3;

    float *input, *output, *kernel;

    size_t bytes_matrix = width * height * sizeof(float);
    size_t bytes_kernel = size * size * sizeof(float);

    cudaMallocManaged(&input, bytes_matrix);
    cudaMallocManaged(&output, bytes_matrix);
    cudaMallocManaged(&kernel, bytes_kernel);

    // fill(input, 1.0f, height*width);

    input[POS(9,0)] = 1.0f;

    // fillRow(input, 2.0f, 0, height, width);
    // fillRow(input, 3.0f, height-1, height, width);

    fill(kernel, 0.1f, size * size);

    print2D(input, height, width);

    cudaMemPrefetchAsync(input, bytes_matrix, GPU);
    cudaMemPrefetchAsync(output, bytes_matrix, GPU);
    cudaMemPrefetchAsync(kernel, bytes_kernel, GPU);

    dim3 grid(((width - 1) / TILE_SIZE) + 1, ((height - 1) / TILE_SIZE) + 1);
    dim3 block(TILE_SIZE, TILE_SIZE);
    size_t memSize = (TILE_SIZE + size - 1) * (TILE_SIZE + size - 1) * sizeof(float);

    printf("Grid: %i, %i, %i\n", grid.x, grid.y, grid.z);
    printf("Block: %i, %i, %i\n", block.x, block.y, block.z);
    printf("Mem: %i x %i (%zu)\n\n", (TILE_SIZE + size - 1), (TILE_SIZE + size - 1), memSize);

    convolution<<<grid, block, memSize>>>(input, kernel, output, height, width, size);

    checkCudaError(cudaDeviceSynchronize());

    print2D(output, height, width);

    cudaFree(input);
    cudaFree(output);
    cudaFree(kernel);

    return 0;
} */
