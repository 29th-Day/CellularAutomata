#pragma once

#include <chrono>
#include <iostream>
#include <iomanip>

/**
 * @brief Time a function call using std::chrono
 *
 */
#define timeit(fn)                                          \
    auto t1 = std::chrono::high_resolution_clock::now();    \
    fn;                                                     \
    auto t2 = std::chrono::high_resolution_clock::now();    \
    std::chrono::duration<double, std::milli> ms = t2 - t1; \
    std::cout << std::left << std::setw(6) << std::setfill('0') << ms.count() << " ms" << std::endl;
