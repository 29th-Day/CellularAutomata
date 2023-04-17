MESSAGE("BUILD EXAMPLE")

set(TARGET example)

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

# link SDL
target_link_libraries(${TARGET}
    PRIVATE
    $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
    $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
)

# link CellularAutomata
target_link_libraries(${TARGET} PRIVATE ${PROJECT_NAME})
