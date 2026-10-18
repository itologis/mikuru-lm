
#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

constexpr std::size_t CONTEXT_SIZE = 4;
constexpr std::size_t HIDDEN_SIZE = 32;

using TokenId = std::size_t;

struct Model {
    std::size_t vocab_size = 0;
    std::vector<double> w1;
    std::vector<double> b1;
    std::vector<double> w2;
    std::vector<double> b2;
};

std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string word;

    for (unsigned char c : text) {
        if (std::isalnum(c) || c == '\'') {
            word += static_cast<char>(std::tolower(c));
        } else {
            if (!word.empty()) {
                tokens.push_back(word);
                word.clear();
            }

            if (c == '.' || c == ',' || c == '!' ||
                c == '?' || c == ';' || c == ':') {
                tokens.emplace_back(
                    1, static_cast<char>(c)
                );
            }
        }
    }

    if (!word.empty()) {
        tokens.push_back(word);
    }

    return tokens;
}

std::string format_text(const std::vector<std::string>& tokens) {
    std::string result;
    bool capitalize_next = true;

    for (const auto& token : tokens) {
        const bool punctuation =
            token == "." || token == "," || token == "!" ||
            token == "?" || token == ";" || token == ":";

        if (!result.empty() && !punctuation) {
            result += ' ';
        }

        std::string formatted = token;

        if (capitalize_next && !punctuation && !formatted.empty()) {
            formatted[0] = static_cast<char>(
                std::toupper(
                    static_cast<unsigned char>(formatted[0])
                )
            );

            capitalize_next = false;
        }

        result += formatted;

        if (token == "." || token == "!" || token == "?") {
            capitalize_next = true;
        }
    }

    return result;
}

bool load_vocabulary(
    std::vector<std::string>& id_to_token,
    std::unordered_map<std::string, TokenId>& token_to_id
) {
    std::ifstream file("data/vocab.txt");

    if (!file) {
        std::cerr << "Could not open data/vocab.txt\n";
        return false;
    }

    std::string token;

    while (std::getline(file, token)) {
        if (token.empty() ||
            token_to_id.find(token) != token_to_id.end()) {
            std::cerr << "Invalid or duplicate vocabulary token.\n";
            return false;
        }

        const TokenId id = id_to_token.size();

        id_to_token.push_back(token);
        token_to_id.emplace(token, id);
    }

    if (file.bad()) {
        std::cerr << "Error reading vocabulary.\n";
        return false;
    }

    const std::vector<std::string> required = {
        "<PAD>", "<UNK>", "<BOS>", "<EOS>"
    };

    if (id_to_token.size() < required.size()) {
        std::cerr << "Vocabulary is missing special tokens.\n";
        return false;
    }

    for (std::size_t i = 0; i < required.size(); ++i) {
        if (id_to_token[i] != required[i]) {
            std::cerr << "Special tokens are in the wrong order.\n";
            return false;
        }
    }

    return true;
}

bool load_model(const std::string& path, Model& model) {
    std::ifstream file(path);

    if (!file) {
        std::cerr << "Could not open model: " << path << '\n';
        return false;
    }

    std::string magic;
    std::size_t saved_context_size;
    std::size_t saved_hidden_size;

    if (!(file >> magic) || magic != "MIKURU_MLP_V1" ||
        !(file >> model.vocab_size
               >> saved_context_size
               >> saved_hidden_size)) {
        std::cerr << "Invalid model file.\n";
        return false;
    }

    if (saved_context_size != CONTEXT_SIZE ||
        saved_hidden_size != HIDDEN_SIZE) {
        std::cerr << "Model architecture does not match this program.\n";
        return false;
    }

    const std::size_t vocab = model.vocab_size;

    auto read_values = [&file](
        std::vector<double>& values,
        std::size_t count
    ) {
        values.resize(count);

        for (double& value : values) {
            if (!(file >> value) || !std::isfinite(value)) {
                return false;
            }
        }

        return true;
    };

    if (!read_values(model.b1, HIDDEN_SIZE) ||
        !read_values(model.w1, CONTEXT_SIZE * vocab * HIDDEN_SIZE) ||
        !read_values(model.b2, vocab) ||
        !read_values(model.w2, HIDDEN_SIZE * vocab)) {
        std::cerr << "Incomplete or invalid model parameters.\n";
        return false;
    }

    return true;
}

std::vector<double> predict(
    const std::array<TokenId, CONTEXT_SIZE>& context,
    const Model& model
) {
    std::vector<double> hidden(HIDDEN_SIZE, 0.0);

    // Calculate the hidden layer.
    for (std::size_t j = 0; j < HIDDEN_SIZE; ++j) {
        double sum = model.b1[j];

        for (std::size_t position = 0;
             position < CONTEXT_SIZE;
             ++position) {

            const std::size_t index =
                (position * model.vocab_size + context[position])
                * HIDDEN_SIZE + j;

            sum += model.w1[index];
        }

        hidden[j] = std::tanh(sum);
    }

    // Calculate a score for every possible next token.
    std::vector<double> logits(model.vocab_size);

    for (std::size_t token = 0;
         token < model.vocab_size;
         ++token) {

        double sum = model.b2[token];

        for (std::size_t j = 0; j < HIDDEN_SIZE; ++j) {
            sum += hidden[j]
                   * model.w2[j * model.vocab_size + token];
        }

        logits[token] = sum;
    }

    // Softmax converts the scores into probabilities.
    const double max_logit =
        *std::max_element(logits.begin(), logits.end());

    double total = 0.0;

    for (double& logit : logits) {
        logit = std::exp(logit - max_logit);
        total += logit;
    }

    for (double& probability : logits) {
        probability /= total;
    }

    return logits;
}

TokenId sample_token(
    std::vector<double> probabilities,
    std::mt19937& rng
) {
    // These are structural tokens, not normal generated words.
    probabilities[0] = 0.0; // <PAD>
    probabilities[1] = 0.0; // <UNK>
    probabilities[2] = 0.0; // <BOS>

    // Temperature below 1 makes choices more focused.
    constexpr double temperature = 0.8;

    double total = 0.0;

    for (double& probability : probabilities) {
        probability = std::pow(probability, 1.0 / temperature);
        total += probability;
    }

    if (total <= 0.0) {
        return 3; // <EOS>
    }

    std::uniform_real_distribution<double> choose(0.0, total);
    const double selected = choose(rng);

    double cumulative = 0.0;

    for (TokenId id = 0; id < probabilities.size(); ++id) {
        cumulative += probabilities[id];

        if (selected < cumulative) {
            return id;
        }
    }

    return 3; // <EOS>
}

int main() {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, TokenId> token_to_id;

    if (!load_vocabulary(id_to_token, token_to_id)) {
        return 1;
    }

    Model model;

    if (!load_model("data/mikuru_lm.weights", model)) {
        return 1;
    }

    if (model.vocab_size != id_to_token.size()) {
        std::cerr << "Model and vocabulary sizes do not match.\n";
        return 1;
    }

    const TokenId pad_id = token_to_id.at("<PAD>");
    const TokenId unk_id = token_to_id.at("<UNK>");
    const TokenId bos_id = token_to_id.at("<BOS>");
    const TokenId eos_id = token_to_id.at("<EOS>");

    std::mt19937 rng(std::random_device{}());

    std::cout << "Mikuru's neural text generator is ready!\n";
    std::cout << "Enter seed text, leave it blank to start a sentence,\n";
    std::cout << "or type q to quit.\n";
    std::cout << "Note: she continues text; she does not yet answer questions.\n";

    std::string input;

    while (true) {
        std::cout << "\nSeed text> ";

        if (!std::getline(std::cin, input)) {
            break;
        }

        if (input == "q" || input == "quit") {
            break;
        }

        auto seed_tokens = tokenize(input);

        // Start with BOS, followed by the supplied seed text.
        std::vector<TokenId> sequence = {bos_id};

        for (const auto& token : seed_tokens) {
            auto it = token_to_id.find(token);

            sequence.push_back(
                it == token_to_id.end() ? unk_id : it->second
            );
        }

        // Use the final four IDs as the neural network's context.
        std::array<TokenId, CONTEXT_SIZE> context;
        context.fill(pad_id);

        const std::size_t available =
            std::min(CONTEXT_SIZE, sequence.size());

        for (std::size_t i = 0; i < available; ++i) {
            context[CONTEXT_SIZE - available + i] =
                sequence[sequence.size() - available + i];
        }

        // Keep the seed in the displayed output.
        std::vector<std::string> response = seed_tokens;

        // Generate no more than 30 tokens for each seed.
        for (int step = 0; step < 30; ++step) {
            auto probabilities = predict(context, model);

            const TokenId next = sample_token(
                std::move(probabilities), rng
            );

            if (next == eos_id) {
                break;
            }

            response.push_back(id_to_token[next]);

            // Slide the context forward by one token.
            for (std::size_t i = 0; i + 1 < CONTEXT_SIZE; ++i) {
                context[i] = context[i + 1];
            }

            context[CONTEXT_SIZE - 1] = next;
        }

        std::cout << "Mikuru> " << format_text(response) << '\n';
    }

    std::cout << "Mikuru's session has ended.\n";

    return 0;
}
