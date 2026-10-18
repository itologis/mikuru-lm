
#include <cctype>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

using TokenId = std::size_t;

const std::string VOCAB_PATH = "data/vocab.txt";

const std::vector<std::string> SPECIAL_TOKENS = {
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>"
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

void add_token(
    const std::string& token,
    std::unordered_map<std::string, TokenId>& token_to_id,
    std::vector<std::string>& id_to_token
) {
    if (token_to_id.find(token) != token_to_id.end()) {
        return;
    }

    const TokenId id = id_to_token.size();

    token_to_id[token] = id;
    id_to_token.push_back(token);
}

bool save_vocabulary(
    const std::string& path,
    const std::vector<std::string>& id_to_token
) {
    std::ofstream file(path);

    if (!file) {
        std::cerr << "Could not save vocabulary: "
                  << path << '\n';
        return false;
    }

    // Save in ID order, not unordered_map iteration order.
    for (const auto& token : id_to_token) {
        file << token << '\n';
    }

    if (!file) {
        std::cerr << "Error writing vocabulary file.\n";
        return false;
    }

    return true;
}

bool load_vocabulary(
    const std::string& path,
    std::unordered_map<std::string, TokenId>& token_to_id,
    std::vector<std::string>& id_to_token
) {
    std::ifstream file(path);

    if (!file) {
        return false;
    }

    std::string token;

    while (std::getline(file, token)) {
        if (token.empty()) {
            std::cerr << "Invalid empty token in vocabulary.\n";
            return false;
        }

        if (token_to_id.find(token) != token_to_id.end()) {
            std::cerr << "Duplicate token in vocabulary: "
                      << token << '\n';
            return false;
        }

        add_token(token, token_to_id, id_to_token);
    }

    if (file.bad()) {
        std::cerr << "Error reading vocabulary file.\n";
        return false;
    }

    if (id_to_token.size() < SPECIAL_TOKENS.size()) {
        std::cerr << "Vocabulary is missing special tokens.\n";
        return false;
    }

    for (std::size_t i = 0; i < SPECIAL_TOKENS.size(); ++i) {
        if (id_to_token[i] != SPECIAL_TOKENS[i]) {
            std::cerr << "Unexpected special-token order.\n";
            return false;
        }
    }

    return true;
}

int main(int argc, char* argv[]) {
    const bool rebuild =
        argc > 1 &&
        std::string(argv[1]) == "--rebuild-vocab";

    // Read the corpus and tokenize its sentences.
    std::ifstream corpus_file("data/mikuru_corpus.txt");

    if (!corpus_file) {
        std::cerr << "Could not open training text.\n";
        return 1;
    }

    std::vector<std::vector<std::string>> corpus;
    std::string line;

    while (std::getline(corpus_file, line)) {
        auto tokens = tokenize(line);

        if (!tokens.empty()) {
            corpus.push_back(tokens);
        }
    }

    if (corpus_file.bad()) {
        std::cerr << "Error reading training text.\n";
        return 1;
    }

    if (corpus.empty()) {
        std::cerr << "No sentences found in training text.\n";
        return 1;
    }

    std::unordered_map<std::string, TokenId> token_to_id;
    std::vector<std::string> id_to_token;

    bool vocabulary_loaded = false;

    // Reuse the existing vocabulary unless rebuilding was requested.
    if (!rebuild) {
        std::ifstream existing_file(VOCAB_PATH);

        if (existing_file) {
            existing_file.close();

            if (!load_vocabulary(
                    VOCAB_PATH, token_to_id, id_to_token)) {
                std::cerr << "Fix the vocabulary file or rebuild it.\n";
                return 1;
            }

            vocabulary_loaded = true;

            std::cout << "Loaded saved vocabulary.\n";
        }
    }

    // Create a new vocabulary on the first run or when requested.
    if (!vocabulary_loaded) {
        for (const auto& token : SPECIAL_TOKENS) {
            add_token(token, token_to_id, id_to_token);
        }

        for (const auto& sentence : corpus) {
            for (const auto& token : sentence) {
                add_token(token, token_to_id, id_to_token);
            }
        }

        if (!save_vocabulary(VOCAB_PATH, id_to_token)) {
            return 1;
        }

        std::cout << "Created and saved vocabulary.\n";
    }

    // Encode sentences using the persistent token IDs.
    const TokenId bos_id = token_to_id.at("<BOS>");
    const TokenId eos_id = token_to_id.at("<EOS>");
    const TokenId unk_id = token_to_id.at("<UNK>");

    std::vector<std::vector<TokenId>> encoded_corpus;

    for (const auto& sentence : corpus) {
        std::vector<TokenId> encoded_sentence;
        encoded_sentence.push_back(bos_id);

        for (const auto& token : sentence) {
            auto it = token_to_id.find(token);

            if (it == token_to_id.end()) {
                // New words not present in the saved vocabulary.
                encoded_sentence.push_back(unk_id);
            } else {
                encoded_sentence.push_back(it->second);
            }
        }

        encoded_sentence.push_back(eos_id);
        encoded_corpus.push_back(encoded_sentence);
    }

    // Display the vocabulary in stable ID order.
    std::cout << "\nMikuru's vocabulary\n";
    std::cout << "Unique tokens: "
              << id_to_token.size() << "\n\n";

    for (TokenId id = 0; id < id_to_token.size(); ++id) {
        std::cout << id << " -> " << id_to_token[id] << '\n';
    }

    std::cout << "\nEncoded training sentences\n";

    for (const auto& sentence : encoded_corpus) {
        for (std::size_t i = 0; i < sentence.size(); ++i) {
            if (i > 0) {
                std::cout << ' ';
            }

            std::cout << sentence[i];
        }

        std::cout << '\n';
    }

    std::cout << "\nVocabulary saved at: "
              << VOCAB_PATH << '\n';

    return 0;
}
