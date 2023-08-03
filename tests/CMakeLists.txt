MESSAGE("BUILD TESTS")

set(TARGET UnitTest)

# packages
find_package(doctest CONFIG REQUIRED)

# includes
include(doctest)

# build executable
add_executable(${TARGET} unittest.cpp)

target_link_libraries(${TARGET} PRIVATE
    # CellularAutomata
    CellularAutomata
    # doctest
    doctest::doctest
)

# run tests
enable_testing()
doctest_discover_tests(${TARGET})

# # Add the /NODEFAULTLIB:LIBCMT linker option
# if(${USE_CUDA} AND ${MSVC})
#   target_link_options(${TARGET} PRIVATE /NODEFAULTLIB:LIBCMT)
# endif()
