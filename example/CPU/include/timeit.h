#pragma once

#include <chrono>
#include <iostream>

#define timeit(fn)                                          \
    auto t1 = std::chrono::high_resolution_clock::now();    \
    fn;                                                     \
    auto t2 = std::chrono::high_resolution_clock::now();    \
    std::chrono::duration<double, std::milli> ms = t2 - t1; \
    std::cout << ms.count() << " ms" << std::endl;
