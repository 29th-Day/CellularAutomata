# Build path - Base / OpenMP

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
add_library(${TARGET} STATIC ${FILES})

# Auto Vectorizer
# target_compile_options(${PROJECT_NAME} PRIVATE /Qpar /Qpar-report:2)

# link OpenMP
if(${OpenMP_CXX_FOUND})
    MESSAGE(STATUS "Using OpenMP (CXX Version ${OpenMP_CXX_VERSION})")
    target_compile_definitions(${TARGET} PRIVATE OPENMP=1)
    target_compile_options(${TARGET} PRIVATE /openmp)
    target_link_libraries(${TARGET} PRIVATE OpenMP::OpenMP_CXX)
endif()

# include header files (public so the others can access it (?))
target_include_directories(${TARGET} PUBLIC include)
