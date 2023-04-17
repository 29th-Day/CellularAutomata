MESSAGE("BUILD EXAMPLE")

set(FILES_SOURCES
    source/main.cpp
    source/display.cpp
)

add_executable(example  ${FILES_SOURCES})

# include header files
target_include_directories(example PRIVATE include)

# link SDL
find_package(SDL2 CONFIG REQUIRED)
target_link_libraries(example
    PRIVATE
    $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
    $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
)

# link CellularAutomata
target_link_libraries(example PRIVATE ${PROJECT_NAME})