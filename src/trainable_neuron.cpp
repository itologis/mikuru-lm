
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

struct Sample {
    double input;
    double target;
};

bool load_samples(
    const std::string& path,
    std::vector<Sample>& samples
) {
    std::ifstream file(path);

    if (!file) {
        std::cerr << "Could not open " << path << '\n';
        return false;
    }

    std::string line;
    int line_number = 0;

    while (std::getline(file, line)) {
        ++line_number;

        if (line.empty()) {
            continue;
        }

        std::stringstream row(line);
        double input;
        double target;
        char comma;

        if (!(row >> input >> comma >> target) || comma != ',') {
            std::cerr << "Invalid data in " << path
                      << " at line " << line_number << '\n';
            return false;
        }

        samples.push_back({input, target});
    }

    if (samples.empty()) {
        std::cerr << "No examples found in " << path << '\n';
        return false;
    }

    return true;
}

bool load_model(
    const std::string& path,
    double& weight,
    double& bias
) {
    std::ifstream file(path);

    if (!file) {
        return false;
    }

    if (!(file >> weight >> bias)) {
        std::cerr << "Could not read saved model: "
                  << path << '\n';
        return false;
    }

    return true;
}

bool save_model(
    const std::string& path,
    double weight,
    double bias
) {
    std::ofstream file(path);

    if (!file) {
        std::cerr << "Could not save model: " << path << '\n';
        return false;
    }

    file << std::setprecision(17)
         << weight << '\n'
         << bias << '\n';

    return static_cast<bool>(file);
}

int main(int argc, char* argv[]) {
    const std::string model_path = "data/mikuru.weights";

    const bool force_training =
        argc > 1 && std::string(argv[1]) == "--train";

    double weight = 0.0;
    double bias = 0.0;

        // Interactive prediction mode.
    if (argc > 1 && std::string(argv[1]) == "--predict") {
        if (!load_model(model_path, weight, bias)) {
            std::cerr << "No saved model found. Train Mikuru first.\n";
            return 1;
        }

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Mikuru is ready! Enter a number, or type q to quit.\n";

        std::string line;

        while (true) {
            std::cout << "\nInput> " << std::flush;

            if (!std::getline(std::cin, line)) {
                break;
            }

            if (line == "q" || line == "quit" || line == "exit") {
                break;
            }

            std::stringstream row(line);
            double input;
            char extra;

            if (!(row >> input) || (row >> extra)) {
                std::cout << "Please enter one valid number.\n";
                continue;
            }

            double prediction = input * weight + bias;

            std::cout << "Mikuru predicts: " << prediction << '\n';
        }

        std::cout << "Prediction session ended.\n";
        return 0;
    }

    std::vector<Sample> test_data;

    if (!load_samples("data/test.csv", test_data)) {
        return 1;
    }

    // Try to reuse the model unless training was requested.
    bool model_loaded =
        !force_training && load_model(model_path, weight, bias);

    if (model_loaded) {
        std::cout << "Loaded saved parameters from "
                  << model_path << "\n\n";
    } else {
        // Load training examples only when training is needed.
        std::vector<Sample> training_data;

        if (!load_samples("data/train.csv", training_data)) {
            return 1;
        }

        // Initialize fresh parameters.
        std::mt19937 generator(std::random_device{}());

        std::uniform_real_distribution<double> initial_value(
            -0.5, 0.5
        );

        weight = initial_value(generator);
        bias = initial_value(generator);

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Starting weight: " << weight << '\n';
        std::cout << "Starting bias: " << bias << "\n\n";

        const double learning_rate = 0.01;
        const int epochs = 1000;

        // Train the neuron.
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (const Sample& sample : training_data) {
                double prediction =
                    sample.input * weight + bias;

                double error = prediction - sample.target;

                weight -= learning_rate
                          * 2.0 * error * sample.input;

                bias -= learning_rate * 2.0 * error;
            }
        }

        if (!save_model(model_path, weight, bias)) {
            return 1;
        }

        std::cout << "Training complete!\n";
        std::cout << "Saved learned parameters to "
                  << model_path << "\n\n";
    }

    std::cout << std::fixed << std::setprecision(3);

    std::cout << "Learned weight: " << weight << '\n';
    std::cout << "Learned bias: " << bias << "\n\n";

    // Evaluate using examples not used for training.
    std::cout << "Testing unseen examples:\n";

    for (const Sample& sample : test_data) {
        double prediction = sample.input * weight + bias;

        std::cout << "Input: " << sample.input
                  << " | Prediction: " << prediction
                  << " | Target: " << sample.target
                  << " | Error: "
                  << std::abs(prediction - sample.target)
                  << '\n';
    }

    return 0;
}
