add_subdirectory(CPU)

if (${USE_CUDA})
    add_subdirectory(CUDA)
endif()
