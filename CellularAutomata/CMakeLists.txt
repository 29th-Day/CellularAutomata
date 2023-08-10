# Define a function to install header files
function(install_interface_headers TARGET)
    get_target_property(INTERFACE_INCLUDE_DIRS ${TARGET} INTERFACE_INCLUDE_DIRECTORIES)
    install(
        DIRECTORY ${INTERFACE_INCLUDE_DIRS}
        DESTINATION .
    )
endfunction()

set(TARGET ${PROJECT_NAME})

add_library(${TARGET} INTERFACE)

target_include_directories(${TARGET} INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Call the function to install headers
install_interface_headers(${TARGET})

install(
    TARGETS ${TARGET}
    EXPORT ${TARGET}
)

install(
    EXPORT ${TARGET}
    FILE ${TARGET}Config.cmake DESTINATION cmake
)
