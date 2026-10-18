#include <iostream>

int main() {
    const double input = 2.0;
    const double weight = 3.0;
    const double bias = 1.0;
    const double output = input * weight + bias;

    std::cout << "Mikuru's first neuron: " << output << '\n';
    return 0;
}
