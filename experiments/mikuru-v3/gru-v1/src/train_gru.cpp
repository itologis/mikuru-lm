#include "gru_model.hpp"

#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace mg = mikuru_gru;
namespace mk = mikuru;

int main(int argc, char** argv) {
    std::string corpus_path = "data/dialogue.tsv";
    std::string output_dir = "build";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--corpus" && i + 1 < argc) corpus_path = argv[++i];
        else if (arg == "--out" && i + 1 < argc) output_dir = argv[++i];
        else {
            std::cerr << "Usage: train_gru [--corpus PATH] [--out DIR]\n";
            return 2;
        }
    }

    std::vector<mk::Dialogue> dialogues;
    if (!mk::load_dialogues(corpus_path, dialogues)) return 1;
    if (dialogues.size() < 10) {
        std::cerr << "Need at least 10 dialogue pairs.\n";
        return 1;
    }

    // Same deterministic pair-level split used by the MLP trainer.
    std::vector<std::size_t> order(dialogues.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 split_rng(73021);
    std::shuffle(order.begin(), order.end(), split_rng);
    const std::size_t val_count =
        std::max<std::size_t>(1, dialogues.size() / 5);

    std::vector<mk::Dialogue> train_dialogues, val_dialogues;
    for (std::size_t i = 0; i < order.size(); ++i) {
        if (i < val_count) val_dialogues.push_back(dialogues[order[i]]);
        else train_dialogues.push_back(dialogues[order[i]]);
    }

    const auto vocab = mk::build_vocabulary(train_dialogues);
    const auto ids = mk::index_vocabulary(vocab);
    const auto hash = mk::vocabulary_fingerprint(vocab);

    std::filesystem::create_directories(output_dir);
    const std::string vocab_path = output_dir + "/gru-vocab.txt";
    const std::string model_path = output_dir + "/gru-model.txt";

    if (!mk::save_vocabulary(vocab_path, vocab)) return 1;

    const auto training = mg::encode_dialogues(train_dialogues, ids);
    const auto validation = mg::encode_dialogues(val_dialogues, ids);

    std::size_t train_targets = 0, val_targets = 0;
    for (const auto& s : training) train_targets += mg::target_count(s, 1);
    for (const auto& s : validation) val_targets += mg::target_count(s, 1);
    if (!train_targets || !val_targets) {
        std::cerr << "Empty training or validation target set.\n";
        return 1;
    }

    std::mt19937 init_rng(20261019);
    mg::Model model(vocab.size(), init_rng);
    mg::Adam adam(model.params.size());
    std::mt19937 rng(20261020);

    const double initial_train = mg::evaluate(model, training);
    const double initial_val = mg::evaluate(model, validation);
    mg::Model best_model = model;
    double best_val = initial_val;
    std::size_t best_epoch = 0;
    int stale_epochs = 0;

    std::cout << std::fixed << std::setprecision(4)
              << "Mikuru GRU V1 trainer\n"
              << "Dialogue pairs: " << dialogues.size() << '\n'
              << "Training pairs: " << train_dialogues.size() << '\n'
              << "Validation pairs: " << val_dialogues.size() << '\n'
              << "Vocabulary: " << vocab.size() << " tokens\n"
              << "Training targets: " << train_targets << '\n'
              << "Validation targets: " << val_targets << '\n'
              << "Architecture: embedding=" << mg::EMBED_DIM
              << ", GRU hidden=" << mg::HIDDEN_DIM << '\n'
              << "Parameters: " << model.params.size() << '\n'
              << "Optimizer: Adam, learning rate=" << mg::LEARNING_RATE << '\n'
              << "Gradient clipping: global norm " << mg::GRAD_CLIP_NORM << '\n'
              << "Output dropout: " << mg::OUTPUT_DROPOUT << '\n'
              << "Initial training loss: " << initial_train << '\n'
              << "Initial validation loss: " << initial_val << '\n';

    std::vector<std::size_t> sample_order(training.size());
    std::iota(sample_order.begin(), sample_order.end(), 0);

    for (int epoch = 1; epoch <= mg::MAX_EPOCHS; ++epoch) {
        std::shuffle(sample_order.begin(), sample_order.end(), rng);

        for (std::size_t start = 0; start < sample_order.size();
             start += mg::BATCH_SIZE) {
            mg::Grad grad(model.params.size());
            std::size_t batch_targets = 0;
            const std::size_t end =
                std::min(start + mg::BATCH_SIZE, sample_order.size());

            for (std::size_t i = start; i < end; ++i) {
                mg::accumulate_sequence_gradient(
                    model, training[sample_order[i]], rng,
                    grad, batch_targets, mg::OUTPUT_DROPOUT);
            }
            mg::apply_adam(model, grad, adam, batch_targets);
        }

        const double train_loss = mg::evaluate(model, training);
        const double val_loss = mg::evaluate(model, validation);

        std::cout << "Epoch " << std::setw(2) << epoch
                  << " | train loss=" << train_loss
                  << " | validation loss=" << val_loss << '\n';

        if (val_loss < best_val) {
            best_val = val_loss;
            best_model = model;
            best_epoch = static_cast<std::size_t>(epoch);
            stale_epochs = 0;
        } else {
            ++stale_epochs;
        }

        if (stale_epochs >= mg::EARLY_STOPPING_PATIENCE) {
            std::cout << "Early stopping: validation loss stopped improving.\n";
            break;
        }
    }

    model = std::move(best_model);
    std::cout << "Best validation epoch: " << best_epoch << '\n'
              << "Initial training loss: " << initial_train << '\n'
              << "Best-checkpoint training loss: " << mg::evaluate(model, training) << '\n'
              << "Initial validation loss: " << initial_val << '\n'
              << "Best validation loss: " << best_val << '\n';

    if (!mg::save_model(model_path, model, hash)) return 1;
    std::cout << "Saved GRU vocabulary: " << vocab_path << '\n'
              << "Saved GRU model: " << model_path << '\n'
              << "Note: validation loss is not a guarantee of coherent dialogue.\n";
    return 0;
}
