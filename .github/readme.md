# Cellular Automata

The project is still in early development.

TODOs until Version 1.0:

- [x] CPU implementation
- [x] CUDA support
- [x] Examples
- [ ] Tests
- [ ] Benchmark
- [ ] More types supported
- [ ] More activation functions
- [ ] >3 kernel sizes?
- [ ] Custom state function
- [ ] Custom kernel function
- [ ] Documentation
- [ ] Testing with clang / g++?

<!-- (Gif?) -->

## TL;DR

CellularAutomata is a simple C++ library for simulating [cellular automata](https://en.wikipedia.org/wiki/Cellular_automaton). It can simulate discrete (like [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)) or continuous spaces using convolutions and activations.

## How it works

<!-- (Write out explanation)
- Space
- Convolution
- Activation
- Similarity to neural networks -->

Every cell in a cellular automaton use the *information of neighboring cells* and a *fixed set of rules* to determine their own next cell state.

Using convolutions and activations 

![alt text](Convolution_GameOfLife.jpeg)

## Build

Software and versions used by myself. Lower versions may be also working.

- Windows 11
- CMake (> 3.20)
- MSVC (> 19.35)

Optional

- CUDA Toolkit (12.1)

## Documentation

Documentation can be found on the [repository wiki page](https://github.com/29th-Day/CellularAutomata/wiki)

## Motive

The goal  is to build a project with C++, CUDA and CMake to deepen my understanding with the whole ecosystem around C++. This most probably is NOT the most efficient implementation. I am deliberately not using many already build functionalities or libraries (especially for CUDA) in favor of programming out implementations myself (this may change in the future).

## Further information

- [EmergentGarden](https://www.youtube.com/@EmergentGarden/videos): The Life Engine
