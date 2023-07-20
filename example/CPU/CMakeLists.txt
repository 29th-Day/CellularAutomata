MESSAGE("BUILD EXAMPLE (CPU)")

set(TARGET example_cpu)

# packages
find_package(SDL2 CONFIG REQUIRED)

# source files
set(FILES
    source/main.cpp
    source/display.cpp
)

# build executable
add_executable(${TARGET} ${FILES})

# include header files
target_include_directories(${TARGET} PRIVATE include)

target_link_libraries(${TARGET}
    PRIVATE
    # CellularAutomata
    ${PROJECT_NAME}
    # SDL
    $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
    $<IF:$<TARGET_EXISTS:SDL2::SDL2>, SDL2::SDL2, SDL2::SDL2-static>
)
