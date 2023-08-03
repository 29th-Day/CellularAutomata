MESSAGE("BUILD EXAMPLE (CPU)")

set(TARGET example_cpu)

find_package(SDL2 CONFIG REQUIRED)

# source files
set(FILES
    source/main.cpp
    source/display.cpp
)

add_executable(${TARGET} ${FILES})

target_include_directories(${TARGET} PRIVATE include)

target_link_libraries(${TARGET}
    PRIVATE
    # CellularAutomata
    CellularAutomata
    # SDL
    $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
    $<IF:$<TARGET_EXISTS:SDL2::SDL2>, SDL2::SDL2, SDL2::SDL2-static>
    # OpenMP
    $<IF:$<TARGET_EXISTS:OpenMP::OpenMP_CXX>, OpenMP::OpenMP_CXX, >
)
