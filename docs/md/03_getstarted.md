Getting Started {#getstart}
===

This guide provides a step-by-step introduction to using the %CellularAutomata library. We'll walk through a minimal example to help you understand the basic concepts and usage.

```cpp
#include <CellularAutomata>

using namespace CellularAutomata;

int main() {
    State<float> state(10, 10, States::normal, Device::CPU);
    Kernel<float> kernel(3, Kernels::ones, Device::CPU);

    Epoch(&state, &kernel, Activations::sin, true);

    return 0;
}
```

Let's break down each line of the example code:

```cpp
#include <CellularAutomata>
```

This line is the standard include statement. The library has a top-level pseudo file that imports all necessary components.

```cpp
using namespace CellularAutomata;
```

This line shortens the namespace to make subsequent code more concise and readable.

```cpp
State<float> state(10, 10, States::normal, Device::CPU);
```

Here, we create a `State` object representing the cellular automaton's state. We specify a grid size of 10x10, initialize the values in the range [0, 1], and allocate the state on the CPU.

```cpp
Kernel<float> kernel(3, Kernels::ones, Device::CPU);
```

This line creates a `Kernel` object that defines the rules for the cellular automaton's evolution. We use a 3x3 matrix kernel initialized with ones, and it's also allocated on the CPU.

```cpp
Epoch(&state, &kernel, Activations::sin, true);
```

The `Epoch` function runs a single iteration of the cellular automaton. It takes pointers to the state and kernel, an activation function (`sin` in this case), and a boolean flag indicating the border behavior (self looping).

In summary, this example demonstrates the fundamental steps to set up and run a basic cellular automaton simulation using the %CellularAutomata library. As you become more familiar with the library, you can explore more advanced configurations and features to create complex simulations.

### Further examples

More complex [examples](https://github.com/29th-Day/CellularAutomata/tree/main/example) are available in the repository.
