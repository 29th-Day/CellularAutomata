MESSAGE("BUILD EXAMPLE")

set(TARGET example)

# packages
find_package(SDL2 CONFIG REQUIRED)
if(${USE_CUDA})
    find_package(CUDAToolkit REQUIRED)
endif()

# source files
set(FILES
    source/main.cpp
    source/display.cpp
)

# build executable
add_executable(${TARGET} ${FILES})

# include header files
target_include_directories(${TARGET} PRIVATE include)

if(${USE_CUDA})
    target_include_directories(${TARGET} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

    target_link_libraries(${TARGET} PRIVATE CUDA::cudart_static)

    target_link_libraries(${TARGET} PRIVATE ${CUDAToolkit_LIBRARIES})
endif()

set_target_properties(${TARGET} PROPERTIES
        CUDA_ARCHITECTURES ${CUDA_ARCH} # defined in main CMake file
        CUDA_STANDARD 17
)

# link SDL
target_link_libraries(${TARGET}
    PRIVATE
    $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
    $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
)

# link CellularAutomata
target_link_libraries(${TARGET} PRIVATE ${PROJECT_NAME})

# Add the /NODEFAULTLIB:LIBCMT linker option
if(${USE_CUDA} AND MSVC)
  target_link_options(${TARGET} PRIVATE /NODEFAULTLIB:LIBCMT)
endif()
