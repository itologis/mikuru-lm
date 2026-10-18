#ifndef MIKURU_RESPONSE_QUALITY_HPP
#define MIKURU_RESPONSE_QUALITY_HPP

#include "dialogue_common.hpp"

#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mikuru_response {

inline bool is_punctuation_token(const std::string& token) {
    return token == "." || token == "," || token == "!" ||
           token == "?" || token == ";" || token == ":";
}

inline bool is_function_word(const std::string& token) {
    static const std::unordered_set<std::string> words = {
        "a", "an", "the", "and", "or", "but", "to", "of", "in",
        "on", "at", "for", "from", "with", "is", "am", "are",
        "was", "were", "be", "been", "being", "it", "its", "i",
        "me", "my", "you", "your", "we", "they", "them", "this",
        "that", "can", "could", "would", "should", "do", "does",
        "did", "have", "has", "had", "how"
    };
    return words.find(token) != words.end();
}

inline bool generated_response_is_usable(
    const std::vector<std::string>& response
) {
    if (response.empty()) return false;

    const auto& reserved = mikuru::special_tokens();
    std::vector<std::string> words;

    for (const auto& token : response) {
        if (std::find(reserved.begin(), reserved.end(), token) !=
            reserved.end()) {
            return false;
        }

        if (is_punctuation_token(token)) continue;

        std::string normalized = token;
        std::transform(
            normalized.begin(), normalized.end(), normalized.begin(),
            [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            }
        );
        words.push_back(std::move(normalized));
    }

    if (words.empty()) return false;

    for (std::size_t i = 1; i < words.size(); ++i) {
        if (words[i] == words[i - 1]) return false;
    }

    if (words.size() >= 9) {
        std::unordered_map<std::string, std::size_t> trigram_counts;
        for (std::size_t i = 0; i + 2 < words.size(); ++i) {
            const std::string trigram =
                words[i] + '\x1f' + words[i + 1] + '\x1f' + words[i + 2];
            if (++trigram_counts[trigram] >= 3) return false;
        }
    }

    // These are clear unfinished fragments, not a general minimum length.
    if (words.size() == 1) {
        static const std::unordered_set<std::string> fragments = {
            "a", "an", "the", "and", "or", "but", "because", "step"
        };
        if (fragments.find(words.front()) != fragments.end()) {
            return false;
        }
    }

    if (words.size() >= 3) {
        const bool all_function_words = std::all_of(
            words.begin(), words.end(), is_function_word
        );
        const bool conventional_short_phrase =
            words == std::vector<std::string>{"how", "are", "you"};
        if (all_function_words && !conventional_short_phrase) {
            return false;
        }
    }

    if (words.size() >= 5) {
        const std::size_t filler_count = static_cast<std::size_t>(
            std::count_if(
                words.begin(), words.end(), is_function_word
            )
        );
        if (filler_count * 10 >= words.size() * 7) return false;
    }

    return true;
}

inline std::string personality_fallback(
    const std::string& prompt,
    std::mt19937& rng
) {
    const auto tokens = mikuru::tokenize(prompt);
    const auto has_any = [&tokens](
        std::initializer_list<const char*> terms
    ) {
        for (const auto& token : tokens) {
            for (const char* term : terms) {
                if (token == term) return true;
            }
        }
        return false;
    };

    std::vector<std::string> choices;

    if (has_any({
            "sad", "grief", "lost", "scared", "worried", "fear",
            "hurt", "exhausted", "illness", "died", "death", "anxious"
        })) {
        choices = {
            "I'm sorry this is a lot to carry. We can take it slowly; "
            "would talking it through or finding one practical next "
            "step help more?",
            "That sounds difficult. We don't have to rush—what would "
            "feel most helpful right now?"
        };
    } else if (has_any({
                   "write", "story", "scene", "poem", "novel",
                   "character", "chapter", "creative", "idea",
                   "imagine", "imaginary", "festival", "invent",
                   "dream", "song", "music", "art", "draw", "paint"
               })) {
        choices = {
            "That idea has a nice spark! What's the mood or moment "
            "you'd like me to focus on?",
            "There's something fun to explore there. Give me one "
            "detail to build around, and we'll shape it together."
        };
    } else if (has_any({
                   "code", "coding", "program", "programming",
                   "bug", "error", "compiler", "compile", "debug",
                   "python", "html", "css"
               })) {
        choices = {
            "Let's make it concrete: share the smallest example and "
            "the exact error, and we'll trace it one step at a time.",
            "I lost the thread in that reply. Show me the relevant "
            "code or error message, and we'll work through it carefully."
        };
    } else {
        choices = {
            "I lost the thread on that one. Could you add a little "
            "context? I'll work through it with you.",
            "That reply came out scrambled. Try phrasing the question "
            "another way, and we'll take another pass."
        };
    }

    std::uniform_int_distribution<std::size_t> pick(
        0, choices.size() - 1
    );
    return choices[pick(rng)];
}

inline std::string format_or_fallback(
    const std::vector<std::string>& response,
    const std::string& prompt,
    std::mt19937& rng
) {
    if (!generated_response_is_usable(response)) {
        return personality_fallback(prompt, rng);
    }
    return mikuru::format_text(response);
}

} // namespace mikuru_response

#endif
