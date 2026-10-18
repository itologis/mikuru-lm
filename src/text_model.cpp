
#include <cctype>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using Transitions =
    std::unordered_map<std::string, std::vector<std::string>>;

const std::string END = "<END>";

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
                tokens.emplace_back(1, static_cast<char>(c));
            }
        }
    }

    if (!word.empty()) {
        tokens.push_back(word);
    }

    return tokens;
}

bool is_word(const std::string& token) {
    return !token.empty() &&
        std::isalnum(static_cast<unsigned char>(token.front()));
}

std::string format_text(const std::vector<std::string>& tokens) {
    std::string result;

    for (const auto& token : tokens) {
        bool punctuation =
            token == "." || token == "," || token == "!" ||
            token == "?" || token == ";" || token == ":";

        if (!result.empty() && !punctuation) {
            result += ' ';
        }

        result += token;
    }

    if (!result.empty()) {
        result[0] = static_cast<char>(
            std::toupper(static_cast<unsigned char>(result[0]))
        );
    }

    return result;
}

int main() {
    std::ifstream file("data/mikuru_corpus.txt");

    if (!file) {
        std::cerr << "Could not open Mikuru's training text.\n";
        return 1;
    }

    Transitions transitions;
    std::vector<std::string> starting_words;
    std::string line;

    // Learn which tokens tend to follow one another.
    while (std::getline(file, line)) {
        auto tokens = tokenize(line);

        if (tokens.empty()) {
            continue;
        }

        starting_words.push_back(tokens.front());

        for (std::size_t i = 0; i + 1 < tokens.size(); ++i) {
            transitions[tokens[i]].push_back(tokens[i + 1]);
        }

        transitions[tokens.back()].push_back(END);
    }

    if (starting_words.empty()) {
        std::cerr << "The training text is empty.\n";
        return 1;
    }

    std::mt19937 rng(std::random_device{}());

    std::cout << "Mikuru's tiny text model is ready!\n";
    std::cout << "Type a message, or type q to quit.\n";

    std::string message;

    while (true) {
        std::cout << "\nYou> ";

        if (!std::getline(std::cin, message)) {
            break;
        }

        if (message == "q" || message == "quit") {
            break;
        }

        auto prompt_tokens = tokenize(message);
        std::string current;

        // Use a familiar word from the message as a starting point.
        for (auto it = prompt_tokens.rbegin();
             it != prompt_tokens.rend(); ++it) {
            if (is_word(*it) && transitions.count(*it)) {
                current = *it;
                break;
            }
        }

        // If no suitable word was found, choose a random start.
        if (current.empty()) {
            std::uniform_int_distribution<std::size_t> choose_start(
                0, starting_words.size() - 1
            );

            current = starting_words[choose_start(rng)];
        }

        std::vector<std::string> response;

        // Generate up to 24 tokens from learned transitions.
        for (int i = 0; i < 24 && current != END; ++i) {
            response.push_back(current);

            const auto& options = transitions.at(current);

            std::uniform_int_distribution<std::size_t> choose_next(
                0, options.size() - 1
            );

            current = options[choose_next(rng)];
        }

        std::cout << "Mikuru> " << format_text(response) << '\n';
    }

    std::cout << "Mikuru's session has ended.\n";
    return 0;
}
