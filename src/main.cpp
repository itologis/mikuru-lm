#include <iostream>

int main() {
    double input = 2.0;
    double weight = 3.0;
    double bias = 1.0;

    double output = input * weight + bias;

    std::cout << "Miku Zero's first neuron: "
              << output << '\n';

    return 0;
}