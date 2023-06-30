MESSAGE("BUILD EXAMPLE")

set(TARGET example_cuda)

# source files
set(FILES
    source/main.cpp
)

# build executable
add_executable(${TARGET} ${FILES})

target_link_libraries(${TARGET}
    PRIVATE
    # CellularAutomata
    ${PROJECT_NAME}
)
