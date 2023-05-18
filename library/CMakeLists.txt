set(TARGET ${PROJECT_NAME})

# USE_CUDA is always stronger

if (${USE_CUDA})
    add_subdirectory(cuda)
else()
    add_subdirectory(base)
endif()
