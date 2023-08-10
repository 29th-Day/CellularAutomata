Installation {#install}
===

### Download

1. **Download**: Obtain the desired version of %CellularAutomata from the <a href="https://github.com/29th-Day/CellularAutomata/releases" target="_blank">release page</a>.
2. **Installation**: Install or extract the downloaded files into your preferred directory.

### Building from Source

While %CellularAutomata doesn't require direct compilation, you can package it using [CPack](https://cmake.org/cmake/help/book/mastering-cmake/chapter/Packaging%20With%20CPack.html) for distribution.

Follow these steps:

1. **Clone the Repository**: Start by cloning the repository from GitHub with the following command:

```
git clone https://github.com/29th-Day/CellularAutomata.git
```

2. **Build with CMake**: Move into the project directory, create a 'build' folder, and run CMake to configure the build:

```
mkdir build
cd build
cmake ..
```

3. **Packaging with CPack**: Execute CPack within the 'build' directory to create a packaged version:

```
cd build
cpack
```

### Linking to Your Project

To integrate %CellularAutomata into your project, follow these steps:

1. **Specify Package Location**: In your CMake configuration, inform CMake where to find the %CellularAutomata package. This can be accomplished either project-wide using the [CMAKE_MODULE_PATH](https://cmake.org/cmake/help/latest/variable/CMAKE_MODULE_PATH.html) variable or directly within a specific command.

```cmake
find_package(CellularAutomata REQUIRED [PATHS "<path-to-CellularAutomata>/cmake"])
```

2. **Link to Your Target**: Finally, link %CellularAutomata to your desired target within your project:

```cmake
target_link_libraries(<TARGET> PRIVATE CellularAutomata)
```

By following these steps, you will successfully install, build, and link %CellularAutomata into your project.

### Without CMake

When you are not using CMake, you can include the library just like any other library.
