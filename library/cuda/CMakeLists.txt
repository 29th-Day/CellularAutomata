# Build path - CUDA

MESSAGE("BUILD LIBRARY")

find_package(CUDAToolkit REQUIRED)

enable_language(CUDA)

# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
set(RTX3060 "86-real")

MESSAGE(STATUS "Using CUDAToolkit (${CUDAToolkit_VERSION})")

target_compile_definitions(${TARGET} PRIVATE CUDA=1)

set(FILES
    sources/...
    sources/...
)

# build static library
add_library(${TARGET} STATIC ${FILES})

set_property(TARGET ${TARGET} PROPERTY CUDA_ARCHITECTURES ${RTX3060})
set_property(TARGET ${TARGET} PROPERTY CUDA_STANDARD 11)

# target_compile_options(${TARGET} PRIVATE /VERBOSE:LIB)
# target_compile_options(${TARGET} PRIVATE /NODEFAULTLIB:LIBCMT)


# https://stackoverflow.com/questions/72780153/linker-warning-when-building-cuda-c-code-in-debug-mode-using-vs2022
# https://learn.microsoft.com/en-us/cpp/error-messages/tool-errors/linker-tools-warning-lnk4098?view=msvc-170

# include header files (public so the others can access it (?))
target_include_directories(${TARGET} PUBLIC include)
