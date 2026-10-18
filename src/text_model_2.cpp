
#include <cctype>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using Context = std::pair<std::string, std::string>;

struct ContextHash {
    std::size_t operator()(const Context& context) const {
        const std::size_t first_hash =
            std::hash<std::string>{}(context.first);

        const std::size_t second_hash =
            std::hash<std::string>{}(context.second);

        return first_hash ^
            (second_hash + 0x9e3779b9 +
             (first_hash << 6) + (first_hash >> 2));
    }
};

using Transitions =
    std::unordered_map<
        Context,
        std::vector<std::string>,
        ContextHash
    >;

const std::string END = "<END>";

// Split text into lowercase words and punctuation.
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

// Format generated tokens into more readable text.
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

int main() {
    // Load Mikuru's training text.
    std::ifstream file("data/mikuru_corpus.txt");

    if (!file) {
        std::cerr << "Could not open training text.\n";
        return 1;
    }

    Transitions transitions;
    std::vector<Context> starting_contexts;
    std::string line;

    // Learn possible next tokens for every pair of tokens.
    while (std::getline(file, line)) {
        auto tokens = tokenize(line);

        if (tokens.size() < 2) {
            continue;
        }

        starting_contexts.emplace_back(
            tokens[0], tokens[1]
        );

        tokens.push_back(END);

        for (std::size_t i = 0; i + 2 < tokens.size(); ++i) {
            Context context{
                tokens[i],
                tokens[i + 1]
            };

            transitions[context].push_back(tokens[i + 2]);
        }
    }

    if (starting_contexts.empty()) {
        std::cerr << "Not enough text to build the model.\n";
        return 1;
    }

    // Build a reverse index for faster word-based lookups.
    std::unordered_map<
        std::string,
        std::vector<Context>
    > by_first;

    for (const auto& entry : transitions) {
        by_first[entry.first.first].push_back(entry.first);
    }

    // Set up random selection.
    std::mt19937 rng(std::random_device{}());

    auto choose_index = [&](std::size_t count) -> std::size_t {
        if (count == 0) {
            throw std::logic_error(
                "Cannot choose from an empty collection."
            );
        }

        std::uniform_int_distribution<std::size_t> choose(
            0, count - 1
        );

        return choose(rng);
    };

    std::cout << "Mikuru's two-word text model is ready!\n";
    std::cout << "Type a message, or q to quit.\n";

    std::string message;

    while (true) {
        std::cout << "\nYou> ";

        if (!std::getline(std::cin, message)) {
            break;
        }

        if (message == "q" || message == "quit") {
            break;
        }

        auto prompt = tokenize(message);

        Context current;
        bool found = false;

        // First, look for a familiar pair in the message.
        for (std::size_t i = prompt.size(); i > 1; --i) {
            Context candidate{
                prompt[i - 2],
                prompt[i - 1]
            };

            if (transitions.count(candidate)) {
                current = candidate;
                found = true;
                break;
            }
        }

        // If no pair matched, look for a familiar single word.
        if (!found) {
            for (auto it = prompt.rbegin();
                 it != prompt.rend() && !found; ++it) {

                auto candidates = by_first.find(*it);

                if (candidates == by_first.end() ||
                    candidates->second.empty()) {
                    continue;
                }

                current = candidates->second[
                    choose_index(candidates->second.size())
                ];

                found = true;
            }
        }

        // If nothing matched, choose a training sentence's start.
        if (!found) {
            current = starting_contexts[
                choose_index(starting_contexts.size())
            ];
        }

        std::vector<std::string> response{
            current.first,
            current.second
        };

        // Generate up to 28 additional tokens.
        for (int i = 0; i < 28; ++i) {
            auto it = transitions.find(current);

            if (it == transitions.end() || it->second.empty()) {
                break;
            }

            const auto& options = it->second;

            const std::string& next = options[
                choose_index(options.size())
            ];

            if (next == END) {
                break;
            }

            response.push_back(next);

            // Shift the context forward by one token.
            current = {
                current.second,
                next
            };
        }

        std::cout << "Mikuru> "
                  << format_text(response)
                  << '\n';
    }

    std::cout << "Mikuru's session has ended.\n";

    return 0;
}
