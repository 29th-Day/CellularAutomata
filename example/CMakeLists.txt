
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(example source/main.cpp source/display.cpp)

find_package(SDL2 CONFIG REQUIRED)
target_link_libraries(example
    PRIVATE
    $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
    $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
)

target_link_libraries(example PRIVATE ${PROJECT_NAME})