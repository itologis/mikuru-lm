
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

using TokenId = std::size_t;

constexpr std::size_t CONTEXT_SIZE = 4;

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

bool load_vocabulary(
    const std::string& path,
    std::unordered_map<std::string, TokenId>& token_to_id,
    std::vector<std::string>& id_to_token
) {
    std::ifstream file(path);

    if (!file) {
        std::cerr << "Could not open vocabulary: "
                  << path << '\n';
        return false;
    }

    std::string token;

    while (std::getline(file, token)) {
        if (token.empty() ||
            token_to_id.find(token) != token_to_id.end()) {
            std::cerr << "Invalid or duplicate vocabulary token.\n";
            return false;
        }

        TokenId id = id_to_token.size();

        token_to_id.emplace(token, id);
        id_to_token.push_back(token);
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
            std::cerr << "Unexpected special-token order.\n";
            return false;
        }
    }

    return true;
}

int main() {
    std::unordered_map<std::string, TokenId> token_to_id;
    std::vector<std::string> id_to_token;

    if (!load_vocabulary(
            "data/vocab.txt",
            token_to_id,
            id_to_token)) {
        return 1;
    }

    const TokenId pad_id = token_to_id.at("<PAD>");
    const TokenId unk_id = token_to_id.at("<UNK>");
    const TokenId bos_id = token_to_id.at("<BOS>");
    const TokenId eos_id = token_to_id.at("<EOS>");

    std::ifstream corpus("data/mikuru_corpus.txt");

    if (!corpus) {
        std::cerr << "Could not open training corpus.\n";
        return 1;
    }

    std::ofstream output("data/training_pairs.csv");

    if (!output) {
        std::cerr << "Could not create training dataset.\n";
        return 1;
    }

    output << "c0,c1,c2,c3,target\n";

    std::string line;
    std::size_t example_count = 0;
    std::size_t preview_count = 0;

    std::cout << "Building Mikuru's training dataset...\n";
    std::cout << "Context size: " << CONTEXT_SIZE << " tokens\n\n";

    while (std::getline(corpus, line)) {
        auto words = tokenize(line);

        if (words.empty()) {
            continue;
        }

        // Convert this sentence into its saved token IDs.
        std::vector<TokenId> sequence;
        sequence.reserve(words.size() + 2);
        sequence.push_back(bos_id);

        for (const auto& word : words) {
            auto it = token_to_id.find(word);

            sequence.push_back(
                it == token_to_id.end() ? unk_id : it->second
            );
        }

        sequence.push_back(eos_id);

        // Each position becomes a next-token prediction example.
        for (std::size_t target_index = 1;
             target_index < sequence.size();
             ++target_index) {

            std::vector<TokenId> context(
                CONTEXT_SIZE, pad_id
            );

            const std::size_t available = std::min(
                CONTEXT_SIZE, target_index
            );

            // Right-align the preceding tokens in the context.
            for (std::size_t j = 0; j < available; ++j) {
                context[CONTEXT_SIZE - available + j] =
                    sequence[target_index - available + j];
            }

            const TokenId target = sequence[target_index];

            // Save numeric IDs as a CSV row.
            for (std::size_t j = 0; j < CONTEXT_SIZE; ++j) {
                if (j > 0) {
                    output << ',';
                }

                output << context[j];
            }

            output << ',' << target << '\n';

            // Display the first few examples for verification.
            if (preview_count < 8) {
                std::cout << "Context: ";

                for (TokenId id : context) {
                    std::cout << id << ' ';
                }

                std::cout << " -> Target: " << target
                          << " (" << id_to_token[target] << ")\n";

                ++preview_count;
            }

            ++example_count;
        }
    }

    if (corpus.bad() || !output) {
        std::cerr << "Error reading or writing dataset.\n";
        return 1;
    }

    if (example_count == 0) {
        std::cerr << "No training examples were generated.\n";
        return 1;
    }

    std::cout << "\nDataset created successfully!\n";
    std::cout << "Training examples: " << example_count << '\n';
    std::cout << "Saved to: data/training_pairs.csv\n";

    return 0;
}
