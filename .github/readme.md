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

Every cell in a cellular automaton use the *information of neighboring cells* and a *fixed set of rules* to determine their own next cell state.

This behavior can be implemented using convolutions and activations. The main advantage of this approach is every set of rules should be able to be implemented using these building blocks.

### Example: Conway's Game of Life

In Game of Life (GoL) the next cell state depends in the Moore-Neighborhood of cells. This is done by using a 3x3 kernel for convolution. Since GoL is a binary system, the kernel values can be defined as following:

![Moore-Neighborhood * Kernel](Convolution_GameOfLife.jpeg)

Each cell has 8 dead or alive neighbors and can be either alive or dead. A minimum of 5 bits (in practice 1 byte) is needed to encode every single combination possible. The lower nibble stores the number of alive neighbors and the higher nibble the current cell state.

In combination with the following activation function every rule of GoL is applied.

```
fn life(int x)
{
    int neighbors = x & 0xF; // lower nibble  - 0b00001111
    bool alive = x & 0xF0;   // higher nibble - 0b11110000

    switch(neighbors)
    {
        case 3:
            return 1;
        case 2:
            return (alive) ? 1 : 0;
        default:
            return 0;
    }
}
```

The rules of GoL can be found on the web, e.g. [Wikipedia](https://en.wikipedia.org/wiki/Conway's_Game_of_Life#Rules)

### Side note

This process is exactly what convolutional neural networks do to extract relevant information from images. The kernel values are changed during training to improve the output of the convolution.

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

- [EmergentGarden](https://www.youtube.com/@EmergentGarden/videos): Life Engine-Series
- 3Blue1Brown: [But what is a convolution?](https://youtu.be/KuXjwB4LzSA)
- Mordvintsev, et al., "[Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/)", Distill, 2020
