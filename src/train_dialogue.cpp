#include "dialogue_common.hpp"

int main() {
    using namespace mikuru;
    const std::string corpus_path = "data/dialogue.tsv";
    const std::string build_dir = "build";
    const std::string vocab_path = build_dir + "/dialogue-vocab.txt";
    const std::string model_path = build_dir + "/mikuru-dialogue.weights";

    std::vector<Dialogue> dialogues;
    if (!load_dialogues(corpus_path, dialogues)) {
        std::cerr << "No valid dialogue pairs loaded.\n";
        return 1;
    }
    if (dialogues.size() < 10) {
        std::cerr << "Need at least 10 dialogue pairs for a useful smoke training run.\n";
        return 1;
    }

    // Stable, reproducible split before constructing the vocabulary.
    std::vector<std::size_t> order(dialogues.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 split_rng(73021);
    std::shuffle(order.begin(), order.end(), split_rng);
    const std::size_t validation_count = std::max<std::size_t>(1, dialogues.size() / 5);

    std::vector<Dialogue> training_dialogues;
    std::vector<Dialogue> validation_dialogues;
    for (std::size_t position = 0; position < order.size(); ++position) {
        if (position < validation_count) validation_dialogues.push_back(dialogues[order[position]]);
        else training_dialogues.push_back(dialogues[order[position]]);
    }

    const auto vocab = build_vocabulary(training_dialogues);
    const auto ids = index_vocabulary(vocab);
    const auto vocab_hash = vocabulary_fingerprint(vocab);
    if (!std::filesystem::create_directories(build_dir) && !std::filesystem::exists(build_dir)) {
        std::cerr << "Could not create build directory.\n";
        return 1;
    }
    if (!save_vocabulary(vocab_path, vocab)) return 1;

    const auto training_samples = make_samples(training_dialogues, ids);
    const auto validation_samples = make_samples(validation_dialogues, ids);
    if (training_samples.empty() || validation_samples.empty()) {
        std::cerr << "Training/validation split produced no samples.\n";
        return 1;
    }

    Network network(vocab.size());
    std::mt19937 rng(20261018);
    std::uniform_real_distribution<double> init(-0.08, 0.08);
    for (double& value : network.w1) value = init(rng);
    for (double& value : network.w2) value = init(rng);

    const double initial_training_loss = loss(training_samples, network);
    const double initial_validation_loss = loss(validation_samples, network);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Mikuru dialogue model trainer\n"
              << "Dialogue pairs: " << dialogues.size() << "\n"
              << "Training pairs: " << training_dialogues.size() << "\n"
              << "Validation pairs: " << validation_dialogues.size() << "\n"
              << "Vocabulary: " << vocab.size() << " tokens\n"
              << "Training targets: " << training_samples.size() << "\n"
              << "Validation targets: " << validation_samples.size() << "\n"
              << "Architecture: context=" << CONTEXT_SIZE
              << ", hidden=" << HIDDEN_SIZE << "\n"
              << "Initial training loss: " << initial_training_loss << "\n"
              << "Initial validation loss: " << initial_validation_loss << "\n";

    std::vector<std::size_t> sample_order(training_samples.size());
    std::iota(sample_order.begin(), sample_order.end(), 0);

    Network best_network = network;
    double best_validation_loss = initial_validation_loss;
    std::size_t best_epoch = 0;
    int epochs_without_improvement = 0;
    constexpr int EARLY_STOPPING_PATIENCE = 4;

    for (int epoch = 1; epoch <= EPOCHS; ++epoch) {
        std::shuffle(sample_order.begin(), sample_order.end(), rng);
        for (const auto sample_index : sample_order) {
            const Sample& sample = training_samples[sample_index];
            const ForwardPass pass = forward(sample, network);

            std::vector<double> output_gradient = pass.probabilities;
            output_gradient[sample.target] -= 1.0;

            std::vector<double> hidden_gradient(HIDDEN_SIZE, 0.0);
            for (std::size_t j = 0; j < HIDDEN_SIZE; ++j) {
                for (std::size_t token = 0; token < vocab.size(); ++token) {
                    hidden_gradient[j] +=
                        network.w2[j * vocab.size() + token] * output_gradient[token];
                }
            }

            // Update output layer after hidden gradients have been computed.
            for (std::size_t j = 0; j < HIDDEN_SIZE; ++j) {
                for (std::size_t token = 0; token < vocab.size(); ++token) {
                    const std::size_t index = j * vocab.size() + token;
                    network.w2[index] -= LEARNING_RATE * pass.hidden[j] * output_gradient[token];
                }
            }
            for (std::size_t token = 0; token < vocab.size(); ++token) {
                network.b2[token] -= LEARNING_RATE * output_gradient[token];
            }

            // Backpropagate through tanh and the positional token lookup.
            for (std::size_t j = 0; j < HIDDEN_SIZE; ++j) {
                const double h = pass.hidden[j];
                const double gradient = hidden_gradient[j] * (1.0 - h * h);
                network.b1[j] -= LEARNING_RATE * gradient;
                for (std::size_t position = 0; position < CONTEXT_SIZE; ++position) {
                    const auto index = w1_index(position, sample.context[position], j, vocab.size());
                    network.w1[index] -= LEARNING_RATE * gradient;
                }
            }
        }
        const double current_validation_loss = loss(validation_samples, network);
        if (current_validation_loss < best_validation_loss) {
            best_validation_loss = current_validation_loss;
            best_network = network;
            best_epoch = static_cast<std::size_t>(epoch);
            epochs_without_improvement = 0;
        } else {
            ++epochs_without_improvement;
        }

        if (epoch == 1 || epoch % 3 == 0 || epoch == EPOCHS) {
            std::cout << "Epoch " << std::setw(2) << epoch
                      << " | train loss=" << loss(training_samples, network)
                      << " | validation loss=" << current_validation_loss << '\n';
        }
        if (epochs_without_improvement >= EARLY_STOPPING_PATIENCE) {
            std::cout << "Early stopping after validation loss stopped improving.\n";
            break;
        }
    }

    network = std::move(best_network);
    std::cout << "Best validation epoch: " << best_epoch << '\n'
              << "Initial training loss: " << initial_training_loss << '\n'
              << "Best-checkpoint training loss: " << loss(training_samples, network) << '\n'
              << "Initial validation loss: " << initial_validation_loss << '\n'
              << "Best validation loss: " << best_validation_loss << '\n';

    if (!save_model(model_path, network, vocab_hash)) return 1;
    std::cout << "Saved vocabulary: " << vocab_path << '\n'
              << "Saved model: " << model_path << '\n';
    std::cout << "Note: these metrics evaluate a tiny dataset, not general language ability.\n";
    return 0;
}
