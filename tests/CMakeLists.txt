MESSAGE("BUILD TESTS")

set(TARGET UnitTest)

# packages
find_package(doctest CONFIG REQUIRED)

# includes
include(doctest)

# build executable
add_executable(${TARGET} unittest.cpp)

# link Doctest
target_link_libraries(${TARGET} PRIVATE doctest::doctest)

# link CellularAutomata
target_link_libraries(${TARGET} PRIVATE ${PROJECT_NAME})

# run tests
enable_testing()
doctest_discover_tests(${TARGET})

# # Add the /NODEFAULTLIB:LIBCMT linker option
# if(${USE_CUDA} AND ${MSVC})
#   target_link_options(${TARGET} PRIVATE /NODEFAULTLIB:LIBCMT)
# endif()
