
#include <cmath>
#include <iomanip>
#include <iostream>

struct Sample {
    double x1;
    double x2;
    double target;
};

// Squashes a value into the range 0 to 1.
double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

int main() {
    const Sample samples[] = {
        {0.0, 0.0, 0.0},
        {0.0, 1.0, 1.0},
        {1.0, 0.0, 1.0},
        {1.0, 1.0, 0.0}
    };

    // Weights and biases for the two hidden neurons.
    double w11 = 0.15, w21 = -0.20, b1 = 0.10;
    double w12 = -0.10, w22 = 0.20, b2 = -0.15;

    // Weights and bias for the output neuron.
    double v1 = 0.20, v2 = -0.30, bo = 0.05;

    const double learning_rate = 0.5;
    const int epochs = 50000;

    // Train the network using backpropagation.
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (const Sample& sample : samples) {
            // Forward pass: hidden layer.
            const double h1 = sigmoid(
                sample.x1 * w11 + sample.x2 * w21 + b1
            );

            const double h2 = sigmoid(
                sample.x1 * w12 + sample.x2 * w22 + b2
            );

            // Forward pass: output layer.
            const double output = sigmoid(
                h1 * v1 + h2 * v2 + bo
            );

            // Calculate how the output should change.
            const double output_delta =
                (sample.target - output) * output * (1.0 - output);

            // Send the learning signal back to the hidden layer.
            const double hidden_delta1 =
                h1 * (1.0 - h1) * output_delta * v1;

            const double hidden_delta2 =
                h2 * (1.0 - h2) * output_delta * v2;

            // Update output neuron parameters.
            v1 += learning_rate * output_delta * h1;
            v2 += learning_rate * output_delta * h2;
            bo += learning_rate * output_delta;

            // Update hidden neuron parameters.
            w11 += learning_rate * hidden_delta1 * sample.x1;
            w21 += learning_rate * hidden_delta1 * sample.x2;
            b1 += learning_rate * hidden_delta1;

            w12 += learning_rate * hidden_delta2 * sample.x1;
            w22 += learning_rate * hidden_delta2 * sample.x2;
            b2 += learning_rate * hidden_delta2;
        }
    }

    // Test all four XOR examples.
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Mikuru's XOR network after training:\n";

    for (const Sample& sample : samples) {
        const double h1 = sigmoid(
            sample.x1 * w11 + sample.x2 * w21 + b1
        );

        const double h2 = sigmoid(
            sample.x1 * w12 + sample.x2 * w22 + b2
        );

        const double output = sigmoid(
            h1 * v1 + h2 * v2 + bo
        );

        std::cout << "Input: (" << sample.x1 << ", " << sample.x2 << ")"
                  << " | Prediction: " << output
                  << " | Target: " << sample.target << '\n';
    }

    return 0;
}
