#pragma once

enum Device
{
    CPU,
    CUDA,
};

template <typename T>
struct State
{
    T *curr;
    T *next;
    int height;
    int width;
    Device device;
};

template <typename T>
struct Kernel
{
    T *kernel;
    int size;
    Device device;
};

namespace Activations
{
    enum OpCode
    {
        normal,
        life,
        sigmoid,
        tanh
    };
}

template <typename T>
using stateFunc = void (*)(T *, const int, const int);

template <typename T>
using kernelFunc = void (*)(T *, const int);

template <typename T>
using activationFunc = T (*)(T);

// float, double, long double
#define decimal typename std::enable_if<std::is_floating_point<T>::value, T>::type

// bool, char, wchar, short, int, long, long long (or extended)
#define integer typename std::enable_if<std::is_integral<T>::value, T>::type
