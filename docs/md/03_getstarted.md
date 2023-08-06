Getting started {#getstart}
===

A minimal example is shown here.

~~~{.cpp}
#include <CellularAutomata>

CellularAutomata::State<float> state(10, 10, CellularAutomata::States::normal, CellularAutomata::Device::CPU);
CellularAutomata::Kernel<float> kernel(3, CellularAutomata::Kernels::ones, CellularAutomata::Device::CPU);

CellularAutomata::Epoch(&state, &kernel, CellularAutomata::Activations::sin, true);
~~~

Let's go line by line.

``` #include <CellularAutomata>```

is the standard include statement. The library has a top-level pseudo file which imports everything.

``` using namespace CellularAutomata;```

shortens the namespace for easier use.

``` State<float> state(10, 10, States::normal, Device::CPU);```

creates a state of a cellular automaton. The grid has a size of 10x10 and, is initialized with values in the range [0, 1], and is allocated on the CPU.

``` Kernel<float> kernel(3, 3, Kernels::ones, Device::CPU);```

creates a kernel for applying the rules of the cellular automaton. It is a 3x3 matrix, is initialized with ones, and is allocated on the CPU.

``` Epoch(&state, &kernel, Activations::sin, true);```

runs a single epoch of the cellular automaton. It takes a pointer to the state and kernel together with an activation function and the behavior at the borders. 
