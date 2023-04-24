MESSAGE("BUILD LIBRARY")

# include packages
if(${USE_OPENMP})
    find_package(OpenMP REQUIRED)
endif()

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

# link OpenMP
if(OpenMP_CXX_FOUND)
    MESSAGE("Using OpenMP")
    target_compile_definitions(${PROJECT_NAME} PRIVATE OPENMP=1)
    target_compile_options(${PROJECT_NAME} PRIVATE /openmp)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()

# include header files (public so the others can access it (?))
target_include_directories(${PROJECT_NAME} PUBLIC include)
