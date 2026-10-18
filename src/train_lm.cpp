
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

constexpr std::size_t CONTEXT_SIZE = 4;
constexpr std::size_t HIDDEN_SIZE = 32;
constexpr int EPOCHS = 300;
constexpr double LEARNING_RATE = 0.03;

using TokenId = std::size_t;

struct Sample {
    std::array<TokenId, CONTEXT_SIZE> context;
    TokenId target;
};

struct ForwardPass {
    std::vector<double> hidden;
    std::vector<double> probabilities;
};

struct Network {
    std::size_t vocab_size;

    // Each context position has a separate set of input weights.
    std::vector<double> w1;
    std::vector<double> b1;

    // Hidden layer to output vocabulary scores.
    std::vector<double> w2;
    std::vector<double> b2;

    explicit Network(std::size_t vocab)
        : vocab_size(vocab),
          w1(CONTEXT_SIZE * vocab * HIDDEN_SIZE),
          b1(HIDDEN_SIZE, 0.0),
          w2(HIDDEN_SIZE * vocab, 0.0),
          b2(vocab, 0.0) {}
};

std::size_t w1_index(
    std::size_t position,
    TokenId token,
    std::size_t neuron,
    std::size_t vocab_size
) {
    return (position * vocab_size + token) * HIDDEN_SIZE
           + neuron;
}

ForwardPass forward(
    const Sample& sample,
    const Network& network
) {
    ForwardPass pass;
    pass.hidden.assign(HIDDEN_SIZE, 0.0);

    // Input layer to hidden layer.
    for (std::size_t j = 0; j < HIDDEN_SIZE; ++j) {
        double sum = network.b1[j];

        for (std::size_t position = 0;
             position < CONTEXT_SIZE;
             ++position) {

            sum += network.w1[
                w1_index(
                    position,
                    sample.context[position],
                    j,
                    network.vocab_size
                )
            ];
        }

        pass.hidden[j] = std::tanh(sum);
    }

    // Hidden layer to output scores.
    std::vector<double> logits(network.vocab_size);

    for (std::size_t token = 0;
         token < network.vocab_size;
         ++token) {

        double sum = network.b2[token];

        for (std::size_t j = 0; j < HIDDEN_SIZE; ++j) {
            sum += pass.hidden[j]
                   * network.w2[j * network.vocab_size + token];
        }

        logits[token] = sum;
    }

    // Stable softmax converts scores into probabilities.
    const double max_logit =
        *std::max_element(logits.begin(), logits.end());

    pass.probabilities.resize(network.vocab_size);

    double total = 0.0;

    for (std::size_t i = 0; i < network.vocab_size; ++i) {
        pass.probabilities[i] =
            std::exp(logits[i] - max_logit);

        total += pass.probabilities[i];
    }

    for (double& probability : pass.probabilities) {
        probability /= total;
    }

    return pass;
}

bool load_samples(
    const std::string& path,
    std::size_t vocab_size,
    std::vector<Sample>& samples
) {
    std::ifstream file(path);

    if (!file) {
        std::cerr << "Could not open dataset: "
                  << path << '\n';
        return false;
    }

    std::string line;

    // Skip the CSV header.
    if (!std::getline(file, line)) {
        std::cerr << "Dataset is empty.\n";
        return false;
    }

    std::size_t line_number = 1;

    while (std::getline(file, line)) {
        ++line_number;

        if (line.empty()) {
            continue;
        }

        std::stringstream row(line);
        std::vector<TokenId> values;
        std::string cell;

        while (std::getline(row, cell, ',')) {
            std::stringstream number(cell);
            TokenId value;
            char extra;

            if (!(number >> value) || (number >> extra)) {
                std::cerr << "Invalid number at dataset line "
                          << line_number << '\n';
                return false;
            }

            values.push_back(value);
        }

        if (values.size() != CONTEXT_SIZE + 1) {
            std::cerr << "Expected four inputs and one target at line "
                      << line_number << '\n';
            return false;
        }

        for (TokenId value : values) {
            if (value >= vocab_size) {
                std::cerr << "Token ID out of range at line "
                          << line_number << '\n';
                return false;
            }
        }

        Sample sample{};

        for (std::size_t i = 0; i < CONTEXT_SIZE; ++i) {
            sample.context[i] = values[i];
        }

        sample.target = values[CONTEXT_SIZE];
        samples.push_back(sample);
    }

    if (file.bad()) {
        std::cerr << "Error reading dataset.\n";
        return false;
    }

    if (samples.empty()) {
        std::cerr << "No training examples found.\n";
        return false;
    }

    return true;
}

std::size_t load_vocab_size(const std::string& path) {
    std::ifstream file(path);

    if (!file) {
        std::cerr << "Could not open vocabulary: "
                  << path << '\n';
        return 0;
    }

    std::size_t count = 0;
    std::string token;

    while (std::getline(file, token)) {
        if (token.empty()) {
            std::cerr << "Empty token in vocabulary.\n";
            return 0;
        }

        ++count;
    }

    if (file.bad()) {
        std::cerr << "Error reading vocabulary.\n";
        return 0;
    }

    return count;
}

double calculate_loss(
    const std::vector<Sample>& samples,
    const Network& network
) {
    double total_loss = 0.0;

    for (const Sample& sample : samples) {
        const ForwardPass pass = forward(sample, network);

        const double probability =
            std::max(pass.probabilities[sample.target], 1e-15);

        total_loss -= std::log(probability);
    }

    return total_loss / static_cast<double>(samples.size());
}

double calculate_accuracy(
    const std::vector<Sample>& samples,
    const Network& network
) {
    std::size_t correct = 0;

    for (const Sample& sample : samples) {
        const ForwardPass pass = forward(sample, network);

        const auto predicted = static_cast<TokenId>(
            std::distance(
                pass.probabilities.begin(),
                std::max_element(
                    pass.probabilities.begin(),
                    pass.probabilities.end()
                )
            )
        );

        if (predicted == sample.target) {
            ++correct;
        }
    }

    return 100.0 * static_cast<double>(correct)
           / static_cast<double>(samples.size());
}

bool save_model(
    const std::string& path,
    const Network& network
) {
    std::ofstream file(path);

    if (!file) {
        std::cerr << "Could not save model: " << path << '\n';
        return false;
    }

    file << std::setprecision(17);
    file << "MIKURU_MLP_V1\n";
    file << network.vocab_size << ' '
         << CONTEXT_SIZE << ' '
         << HIDDEN_SIZE << '\n';

    auto write_values = [&file](
        const std::vector<double>& values
    ) {
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (i > 0) {
                file << ' ';
            }

            file << values[i];
        }

        file << '\n';
    };

    write_values(network.b1);
    write_values(network.w1);
    write_values(network.b2);
    write_values(network.w2);

    if (!file) {
        std::cerr << "Error writing model parameters.\n";
        return false;
    }

    return true;
}

int main() {
    const std::size_t vocab_size =
        load_vocab_size("data/vocab.txt");

    if (vocab_size < 4) {
        std::cerr << "Invalid vocabulary.\n";
        return 1;
    }

    std::vector<Sample> samples;

    if (!load_samples(
            "data/training_pairs.csv",
            vocab_size,
            samples)) {
        return 1;
    }

    std::mt19937 rng(std::random_device{}());

    Network network(vocab_size);

    // Randomly initialize all trainable weights.
    std::uniform_real_distribution<double> initial(-0.1, 0.1);

    for (double& value : network.w1) {
        value = initial(rng);
    }

    for (double& value : network.w2) {
        value = initial(rng);
    }

    const double initial_loss = calculate_loss(samples, network);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Mikuru's tiny neural language model\n";
    std::cout << "Vocabulary size: " << vocab_size << '\n';
    std::cout << "Training examples: " << samples.size() << '\n';
    std::cout << "Hidden neurons: " << HIDDEN_SIZE << '\n';
    std::cout << "Initial loss: " << initial_loss << "\n\n";

    // Shuffle examples between epochs.
    std::vector<std::size_t> order(samples.size());

    for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::shuffle(order.begin(), order.end(), rng);

        for (std::size_t sample_index : order) {
            const Sample& sample = samples[sample_index];

            const ForwardPass pass = forward(sample, network);

            // Cross-entropy gradient with softmax.
            std::vector<double> output_gradient =
                pass.probabilities;

            output_gradient[sample.target] -= 1.0;

            // Calculate hidden gradients before changing output weights.
            std::vector<double> hidden_gradient(
                HIDDEN_SIZE, 0.0
            );

            for (std::size_t j = 0; j < HIDDEN_SIZE; ++j) {
                for (std::size_t token = 0;
                     token < vocab_size;
                     ++token) {

                    hidden_gradient[j] +=
                        network.w2[j * vocab_size + token]
                        * output_gradient[token];
                }
            }

            // Update output weights and biases.
            for (std::size_t j = 0; j < HIDDEN_SIZE; ++j) {
                for (std::size_t token = 0;
                     token < vocab_size;
                     ++token) {

                    const std::size_t index =
                        j * vocab_size + token;

                    network.w2[index] -=
                        LEARNING_RATE * pass.hidden[j]
                        * output_gradient[token];
                }
            }

            for (std::size_t token = 0;
                 token < vocab_size;
                 ++token) {

                network.b2[token] -=
                    LEARNING_RATE * output_gradient[token];
            }

            // Backpropagate through tanh and update input weights.
            for (std::size_t j = 0; j < HIDDEN_SIZE; ++j) {
                const double hidden_value = pass.hidden[j];

                const double gradient =
                    hidden_gradient[j]
                    * (1.0 - hidden_value * hidden_value);

                network.b1[j] -= LEARNING_RATE * gradient;

                for (std::size_t position = 0;
                     position < CONTEXT_SIZE;
                     ++position) {

                    const std::size_t index = w1_index(
                        position,
                        sample.context[position],
                        j,
                        vocab_size
                    );

                    network.w1[index] -=
                        LEARNING_RATE * gradient;
                }
            }
        }

        if ((epoch + 1) % 50 == 0) {
            std::cout << "Epoch " << epoch + 1
                      << " | Loss: "
                      << calculate_loss(samples, network)
                      << '\n';
        }
    }

    const double final_loss = calculate_loss(samples, network);
    const double accuracy = calculate_accuracy(samples, network);

    std::cout << "\nTraining finished.\n";
    std::cout << "Initial loss: " << initial_loss << '\n';
    std::cout << "Final loss: " << final_loss << '\n';
    std::cout << "Training accuracy: " << accuracy << "%\n";

    if (!save_model("data/mikuru_lm.weights", network)) {
        return 1;
    }

    std::cout << "Saved model to data/mikuru_lm.weights\n";

    return 0;
}
