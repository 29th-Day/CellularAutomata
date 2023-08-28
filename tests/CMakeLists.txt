MESSAGE("BUILD TESTS")


find_package(doctest CONFIG REQUIRED)
include(doctest)
enable_testing()

function(create_test target files)
    add_executable(${target} ${files})

    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    target_link_libraries(${target} PRIVATE
        CellularAutomata
        doctest::doctest
    )

    doctest_discover_tests(${target})
endfunction()


create_test(test_core tests.cpp)

find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    enable_language(CUDA)

    create_test(test_cuda tests.cu)
endif()
