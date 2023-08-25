#pragma once

#include <stdexcept>
#include <string>


#define DEFINE_EXCEPTION_CLASS(name)                                                        \
class name : public std::exception {                                                        \
    std::string msg;                                                                        \
    public:                                                                                 \
        /** @brief Default message containing name */                                       \
        explicit name() : msg(#name) { }                                                    \
        /** @brief Message containing name and appended custom message */                   \
        explicit name(std::string message) : msg(#name) { msg = msg + " - " + message; }    \
        /** @brief Override for virtual std::exception */                                   \
        const char* what() const noexcept override { return msg.c_str(); }                  \
    }

namespace CellularAutomata
{
    /**
     * @brief Contains all custom exceptions
    */
    namespace exception
    {
        /**
         * @brief Requested device not available
         */
        DEFINE_EXCEPTION_CLASS(DeviceNotAvailable);

        /**
         * @brief Objects are on different devices
         */
        DEFINE_EXCEPTION_CLASS(DevicesUnequal);

        /**
         * @brief Shape of two matrices are not equal
         */
        DEFINE_EXCEPTION_CLASS(ShapesUnequal);

        /**
         * @brief Accessed a value outside the bounds of matrix
         */
        DEFINE_EXCEPTION_CLASS(OutOfBounds);

        /**
         * @brief A CUDA runtime error occurred
         *
         * For additional infomation look at the [NVIDA CUDA documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html)
         */
        DEFINE_EXCEPTION_CLASS(CudaRuntime);
    }
}

#undef DEFINE_EXCEPTION_CLASS
