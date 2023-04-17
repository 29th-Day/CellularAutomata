MESSAGE("BUILD LIBRARY")

# source files
set(FILES
    source/engine.cpp
    source/rng.cpp
    source/activations.cpp
    source/kernels.cpp
    source/states.cpp
)

# build static library
add_library(${PROJECT_NAME} STATIC ${FILES})

# include header files (public so the others can access it (?))
target_include_directories(${PROJECT_NAME} PUBLIC include)
