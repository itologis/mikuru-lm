#ifndef MIKURU_DIALOGUE_COMMON_HPP
#define MIKURU_DIALOGUE_COMMON_HPP

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mikuru {

constexpr std::size_t CONTEXT_SIZE = 8;
constexpr std::size_t HIDDEN_SIZE = 32;
constexpr int EPOCHS = 15;
constexpr double LEARNING_RATE = 0.035;
using TokenId = std::size_t;

inline const std::vector<std::string>& special_tokens() {
    static const std::vector<std::string> tokens = {
        "<PAD>", "<UNK>", "<BOS>", "<EOS>", "<USER>", "<ASSISTANT>"
    };
    return tokens;
}

struct Dialogue {
    std::string user;
    std::string assistant;
};

struct Sample {
    std::array<TokenId, CONTEXT_SIZE> context{};
    TokenId target = 0;
};

struct ForwardPass {
    std::vector<double> hidden;
    std::vector<double> probabilities;
};

struct Network {
    std::size_t vocab_size;
    std::vector<double> w1;
    std::vector<double> b1;
    std::vector<double> w2;
    std::vector<double> b2;

    explicit Network(std::size_t vocab)
        : vocab_size(vocab),
          w1(CONTEXT_SIZE * vocab * HIDDEN_SIZE, 0.0),
          b1(HIDDEN_SIZE, 0.0),
          w2(HIDDEN_SIZE * vocab, 0.0),
          b2(vocab, 0.0) {}
};

inline std::vector<std::string> tokenize(const std::string& text) {
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
            if (c == '.' || c == ',' || c == '!' || c == '?' || c == ';' || c == ':') {
                tokens.emplace_back(1, static_cast<char>(c));
            }
        }
    }
    if (!word.empty()) tokens.push_back(word);
    return tokens;
}

inline bool load_dialogues(const std::string& path, std::vector<Dialogue>& dialogues) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "Could not open dialogue corpus: " << path << '\n';
        return false;
    }
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(file, line)) {
        ++line_number;
        if (line.empty()) continue;
        if (line_number == 1 && line == "user\tassistant") continue;
        const auto separator = line.find('\t');
        if (separator == std::string::npos) {
            std::cerr << "Expected a tab-separated user/assistant pair on line "
                      << line_number << '\n';
            return false;
        }
        Dialogue dialogue{line.substr(0, separator), line.substr(separator + 1)};
        if (dialogue.user.empty() || dialogue.assistant.empty()) {
            std::cerr << "Empty user message or assistant reply on line "
                      << line_number << '\n';
            return false;
        }
        dialogues.push_back(std::move(dialogue));
    }
    if (file.bad()) {
        std::cerr << "Error reading dialogue corpus.\n";
        return false;
    }
    return !dialogues.empty();
}

inline std::vector<std::string> build_vocabulary(const std::vector<Dialogue>& dialogues) {
    std::unordered_set<std::string> words;
    for (const auto& dialogue : dialogues) {
        for (const auto& token : tokenize(dialogue.user)) words.insert(token);
        for (const auto& token : tokenize(dialogue.assistant)) words.insert(token);
    }
    std::vector<std::string> vocab = special_tokens();
    std::vector<std::string> ordinary(words.begin(), words.end());
    std::sort(ordinary.begin(), ordinary.end());
    vocab.insert(vocab.end(), ordinary.begin(), ordinary.end());
    return vocab;
}

inline std::unordered_map<std::string, TokenId> index_vocabulary(
    const std::vector<std::string>& vocab) {
    std::unordered_map<std::string, TokenId> ids;
    ids.reserve(vocab.size());
    for (TokenId i = 0; i < vocab.size(); ++i) ids.emplace(vocab[i], i);
    return ids;
}

inline bool save_vocabulary(const std::string& path, const std::vector<std::string>& vocab) {
    std::ofstream file(path, std::ios::trunc);
    if (!file) {
        std::cerr << "Could not write vocabulary: " << path << '\n';
        return false;
    }
    for (const auto& token : vocab) file << token << '\n';
    return static_cast<bool>(file);
}

inline bool load_vocabulary(const std::string& path, std::vector<std::string>& vocab) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "Could not open vocabulary: " << path << '\n';
        return false;
    }
    std::unordered_set<std::string> seen;
    std::string token;
    while (std::getline(file, token)) {
        if (token.empty() || !seen.insert(token).second) {
            std::cerr << "Vocabulary contains an empty or duplicate token.\n";
            return false;
        }
        vocab.push_back(token);
    }
    if (file.bad() || vocab.size() < special_tokens().size()) {
        std::cerr << "Vocabulary is incomplete or unreadable.\n";
        return false;
    }
    for (std::size_t i = 0; i < special_tokens().size(); ++i) {
        if (vocab[i] != special_tokens()[i]) {
            std::cerr << "Vocabulary special-token order is incompatible.\n";
            return false;
        }
    }
    return true;
}

inline std::uint64_t vocabulary_fingerprint(const std::vector<std::string>& vocab) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const auto& token : vocab) {
        for (unsigned char c : token) {
            hash ^= c;
            hash *= 1099511628211ULL;
        }
        hash ^= 0xFF;
        hash *= 1099511628211ULL;
    }
    return hash;
}

inline TokenId token_id(const std::string& token,
                        const std::unordered_map<std::string, TokenId>& ids) {
    const auto found = ids.find(token);
    return found == ids.end() ? 1 : found->second; // <UNK>
}

inline std::vector<TokenId> encode_tokens(
    const std::string& text,
    const std::unordered_map<std::string, TokenId>& ids) {
    std::vector<TokenId> encoded;
    for (const auto& token : tokenize(text)) encoded.push_back(token_id(token, ids));
    return encoded;
}

inline std::size_t w1_index(std::size_t position, TokenId token,
                            std::size_t neuron, std::size_t vocab_size) {
    return (position * vocab_size + token) * HIDDEN_SIZE + neuron;
}

inline Sample make_sample(const std::vector<TokenId>& sequence, std::size_t target_index) {
    Sample sample;
    sample.context.fill(0); // <PAD>
    const std::size_t available = std::min(CONTEXT_SIZE, target_index);
    for (std::size_t i = 0; i < available; ++i) {
        sample.context[CONTEXT_SIZE - available + i] =
            sequence[target_index - available + i];
    }
    sample.target = sequence[target_index];
    return sample;
}

inline std::vector<Sample> make_samples(
    const std::vector<Dialogue>& dialogues,
    const std::unordered_map<std::string, TokenId>& ids) {
    std::vector<Sample> samples;
    const TokenId bos = ids.at("<BOS>");
    const TokenId eos = ids.at("<EOS>");
    const TokenId user_marker = ids.at("<USER>");
    const TokenId assistant_marker = ids.at("<ASSISTANT>");
    for (const auto& dialogue : dialogues) {
        std::vector<TokenId> sequence{bos, user_marker};
        auto user_tokens = encode_tokens(dialogue.user, ids);
        sequence.insert(sequence.end(), user_tokens.begin(), user_tokens.end());
        sequence.push_back(assistant_marker);
        const std::size_t first_response_target = sequence.size();
        auto response_tokens = encode_tokens(dialogue.assistant, ids);
        sequence.insert(sequence.end(), response_tokens.begin(), response_tokens.end());
        sequence.push_back(eos);
        for (std::size_t target = first_response_target; target < sequence.size(); ++target) {
            samples.push_back(make_sample(sequence, target));
        }
    }
    return samples;
}

inline ForwardPass forward(const Sample& sample, const Network& network) {
    ForwardPass pass;
    pass.hidden.assign(HIDDEN_SIZE, 0.0);
    for (std::size_t j = 0; j < HIDDEN_SIZE; ++j) {
        double sum = network.b1[j];
        for (std::size_t position = 0; position < CONTEXT_SIZE; ++position) {
            sum += network.w1[w1_index(position, sample.context[position], j,
                                       network.vocab_size)];
        }
        pass.hidden[j] = std::tanh(sum);
    }
    std::vector<double> logits(network.vocab_size);
    for (std::size_t token = 0; token < network.vocab_size; ++token) {
        double sum = network.b2[token];
        for (std::size_t j = 0; j < HIDDEN_SIZE; ++j) {
            sum += pass.hidden[j] * network.w2[j * network.vocab_size + token];
        }
        logits[token] = sum;
    }
    const double max_logit = *std::max_element(logits.begin(), logits.end());
    pass.probabilities.resize(network.vocab_size);
    double total = 0.0;
    for (std::size_t i = 0; i < network.vocab_size; ++i) {
        pass.probabilities[i] = std::exp(logits[i] - max_logit);
        total += pass.probabilities[i];
    }
    if (!(total > 0.0) || !std::isfinite(total)) {
        std::fill(pass.probabilities.begin(), pass.probabilities.end(),
                  1.0 / static_cast<double>(network.vocab_size));
    } else {
        for (double& probability : pass.probabilities) probability /= total;
    }
    return pass;
}

inline double loss(const std::vector<Sample>& samples, const Network& network) {
    if (samples.empty()) return 0.0;
    double total = 0.0;
    for (const auto& sample : samples) {
        const auto pass = forward(sample, network);
        total -= std::log(std::max(pass.probabilities[sample.target], 1e-15));
    }
    return total / static_cast<double>(samples.size());
}

inline bool save_model(const std::string& path, const Network& network,
                       std::uint64_t vocab_hash) {
    std::ofstream file(path, std::ios::trunc);
    if (!file) {
        std::cerr << "Could not write model: " << path << '\n';
        return false;
    }
    file << "MIKURU_DIALOGUE_MLP_V1\n";
    file << network.vocab_size << ' ' << CONTEXT_SIZE << ' '
         << HIDDEN_SIZE << ' ' << vocab_hash << '\n';
    file << std::setprecision(17);
    const auto write_values = [&file](const std::vector<double>& values) {
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (i) file << ' ';
            file << values[i];
        }
        file << '\n';
    };
    write_values(network.b1);
    write_values(network.w1);
    write_values(network.b2);
    write_values(network.w2);
    if (!file) {
        std::cerr << "Error while writing model parameters.\n";
        return false;
    }
    return true;
}

inline bool load_model(const std::string& path, Network& network,
                       std::uint64_t expected_vocab_hash) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "Could not open model: " << path << '\n';
        return false;
    }
    std::string magic;
    std::size_t vocab_size = 0, context_size = 0, hidden_size = 0;
    std::uint64_t hash = 0;
    if (!(file >> magic) || magic != "MIKURU_DIALOGUE_MLP_V1" ||
        !(file >> vocab_size >> context_size >> hidden_size >> hash)) {
        std::cerr << "Invalid or unsupported model file.\n";
        return false;
    }
    if (vocab_size != network.vocab_size || context_size != CONTEXT_SIZE ||
        hidden_size != HIDDEN_SIZE || hash != expected_vocab_hash) {
        std::cerr << "Model/vocabulary/architecture mismatch. Retrain the model.\n";
        return false;
    }
    const auto read_values = [&file](std::vector<double>& values) {
        for (double& value : values) {
            if (!(file >> value) || !std::isfinite(value)) return false;
        }
        return true;
    };
    if (!read_values(network.b1) || !read_values(network.w1) ||
        !read_values(network.b2) || !read_values(network.w2)) {
        std::cerr << "Model file is truncated or contains invalid parameters.\n";
        return false;
    }
    return true;
}

inline std::string format_text(const std::vector<std::string>& tokens) {
    std::string result;
    bool capitalize_next = true;
    for (const auto& token : tokens) {
        const bool punctuation = token == "." || token == "," || token == "!" ||
                                 token == "?" || token == ";" || token == ":";
        if (!result.empty() && !punctuation) result += ' ';
        std::string word = token;
        if (capitalize_next && !punctuation && !word.empty()) {
            word[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(word[0])));
            capitalize_next = false;
        }
        result += word;
        if (token == "." || token == "!" || token == "?") capitalize_next = true;
    }
    return result;
}

} // namespace mikuru

#endif
