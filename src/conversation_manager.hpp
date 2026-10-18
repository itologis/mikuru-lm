#ifndef MIKURU_CONVERSATION_MANAGER_HPP
#define MIKURU_CONVERSATION_MANAGER_HPP

#include "dialogue_common.hpp"

#include <algorithm>
#include <cctype>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mikuru_conversation {

class ConversationManager {
public:
    bool respond(const std::string& input, std::string& reply) {
        if (remember_explicit_fact(input, reply)) return true;
        if (answer_fact_question(input, reply)) return true;

        if (is_short_statement(input)) {
            reply = "Got it. Thanks for telling me.";
            return true;
        }

        return false;
    }

    void reset() {
        facts_.clear();
    }

    std::size_t remembered_fact_count() const {
        return facts_.size();
    }

private:
    struct Fact {
        std::string subject;
        std::string value;
        bool first_person_preference = false;
    };

    std::unordered_map<std::string, Fact> facts_;

    static std::string trim(std::string text) {
        const auto first = text.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) return {};

        const auto last = text.find_last_not_of(" \t\r\n");
        text = text.substr(first, last - first + 1);

        while (!text.empty() &&
               (text.back() == '.' || text.back() == '!' ||
                text.back() == '?' || text.back() == ',')) {
            text.pop_back();
        }

        const auto end = text.find_last_not_of(" \t\r\n");
        if (end == std::string::npos) return {};
        text.erase(end + 1);
        return text;
    }

    static std::string lower_ascii(std::string text) {
        std::transform(
            text.begin(), text.end(), text.begin(),
            [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            }
        );
        return text;
    }

    static std::string canonical_word(std::string word) {
        word = lower_ascii(std::move(word));
        if (word == "favourite") return "favorite";
        if (word == "colour") return "color";
        return word;
    }

    static std::vector<std::string> words(const std::string& text) {
        std::vector<std::string> result;

        for (const auto& token : mikuru::tokenize(text)) {
            if (token == "." || token == "," || token == "!" ||
                token == "?" || token == ";" || token == ":") {
                continue;
            }
            result.push_back(canonical_word(token));
        }

        return result;
    }

    static std::string normalized_key(const std::string& text) {
        const auto parts = words(text);
        std::string key;

        for (const auto& part : parts) {
            if (!key.empty()) key.push_back(' ');
            key += part;
        }

        return key;
    }

    static std::string strip_correction_prefixes(std::string text) {
        text = trim(std::move(text));

        const std::vector<std::string> prefixes = {
            "actually", "correction", "well", "wait", "i mean"
        };

        bool changed = true;
        while (changed && !text.empty()) {
            changed = false;
            const std::string lower = lower_ascii(text);

            for (const auto& prefix : prefixes) {
                if (lower.compare(0, prefix.size(), prefix) != 0) {
                    continue;
                }

                if (lower.size() > prefix.size()) {
                    const char next = lower[prefix.size()];
                    if (next != ',' &&
                        std::isspace(static_cast<unsigned char>(next)) == 0) {
                        continue;
                    }
                }

                text.erase(0, prefix.size());

                const auto first = text.find_first_not_of(" \t\r\n,");
                if (first == std::string::npos) return {};
                text.erase(0, first);

                text = trim(std::move(text));
                changed = true;
                break;
            }
        }

        return text;
    }

    bool remember_explicit_fact(
        const std::string& input,
        std::string& reply
    ) {
        const std::string statement = strip_correction_prefixes(input);
        if (statement.empty()) return false;

        const std::string lower = lower_ascii(statement);

        // Facts such as "My favorite color is red."
        if (lower.rfind("my ", 0) == 0) {
            const std::size_t is_pos = lower.find(" is ", 3);

            if (is_pos != std::string::npos) {
                const std::string subject =
                    trim(statement.substr(3, is_pos - 3));
                std::string value =
                    trim(statement.substr(is_pos + 4));

                if (lower_ascii(value).rfind("called ", 0) == 0) {
                    value = trim(value.substr(7));
                }

                const std::string key = normalized_key(subject);
                if (key.empty() || value.empty()) return false;

                facts_[key] = {subject, value, false};
                reply = "Got it. I'll remember your " + subject +
                        " is " + value + " during this chat.";
                return true;
            }
        }

        // Preferences such as "I like rainy mornings."
        for (const std::string relation : {"like", "love", "prefer"}) {
            const std::string prefix = "i " + relation + " ";
            if (lower.rfind(prefix, 0) != 0) continue;

            const std::string value = trim(
                statement.substr(prefix.size())
            );
            if (value.empty()) return false;

            const std::string key = "i " + relation;
            facts_[key] = {relation, value, true};

            reply = "Got it. You " + relation + " " + value +
                    "; I'll keep that in mind during this chat.";
            return true;
        }

        return false;
    }

    bool answer_fact_question(
        const std::string& input,
        std::string& reply
    ) const {
        const auto query_words = words(input);
        if (query_words.empty()) return false;

        const std::string lower = lower_ascii(input);
        const bool has_question_mark =
            input.find('?') != std::string::npos;

        const std::unordered_set<std::string> question_starters = {
            "what", "which", "who", "where", "when", "why",
            "how", "do", "does", "did", "is", "are", "was",
            "were", "can", "could", "tell", "remember"
        };

        const bool starts_like_question =
            question_starters.find(query_words.front()) !=
            question_starters.end();

        if (!has_question_mark && !starts_like_question) {
            return false;
        }

        const std::unordered_set<std::string> query_set(
            query_words.begin(), query_words.end()
        );

        for (const auto& entry : facts_) {
            const Fact& fact = entry.second;
            const auto subject_words = words(fact.subject);

            if (subject_words.empty()) continue;

            std::size_t matched = 0;
            for (const auto& subject_word : subject_words) {
                if (query_set.find(subject_word) != query_set.end()) {
                    ++matched;
                }
            }

            const double ratio =
                static_cast<double>(matched) /
                static_cast<double>(subject_words.size());

            // Allow natural word-order variations, including:
            // "What color did I say was my favorite?"
            if (ratio >= 0.5 ||
                (matched >= 1 && subject_words.size() <= 2)) {
                if (fact.first_person_preference) {
                    reply = "You " + fact.subject + " " +
                            fact.value + ".";
                } else {
                    reply = "Your " + fact.subject + " is " +
                            fact.value + ".";
                }
                return true;
            }
        }

        return false;
    }

    static bool is_short_statement(const std::string& input) {
        const auto tokens = mikuru::tokenize(input);

        if (tokens.empty() || tokens.size() > 9 ||
            input.find('?') != std::string::npos) {
            return false;
        }

        const auto last = input.find_last_not_of(" \t\r\n");
        if (last == std::string::npos ||
            (input[last] != '.' && input[last] != '!')) {
            return false;
        }

        const std::string first = lower_ascii(tokens.front());

        return first == "i" || first == "my" || first == "we" ||
               first == "it" || first == "that" || first == "today";
    }
};

} // namespace mikuru_conversation

#endif
