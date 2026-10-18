#ifndef MIKURU_CONVERSATION_MANAGER_HPP
#define MIKURU_CONVERSATION_MANAGER_HPP

#include "dialogue_common.hpp"

#include <cctype>
#include <regex>
#include <string>
#include <unordered_map>
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

    static std::string normalized_key(const std::string& text) {
        const auto tokens = mikuru::tokenize(text);
        std::string key;
        for (const auto& token : tokens) {
            if (token == "." || token == "," || token == "!" ||
                token == "?" || token == ";" || token == ":") {
                continue;
            }
            if (!key.empty()) key.push_back(' ');
            key += token;
        }
        return key;
    }

    bool remember_explicit_fact(
        const std::string& input,
        std::string& reply
    ) {
        static const std::regex my_fact(
            R"(^\s*my\s+(.+?)\s+is\s+(?:called\s+)?(.+?)\s*[.!?]*\s*$)",
            std::regex_constants::icase
        );
        static const std::regex preference(
            R"(^\s*i\s+(like|love|prefer)\s+(.+?)\s*[.!?]*\s*$)",
            std::regex_constants::icase
        );

        std::smatch match;
        if (std::regex_match(input, match, my_fact)) {
            const std::string subject = trim(match[1].str());
            const std::string value = trim(match[2].str());
            const std::string key = normalized_key(subject);
            if (key.empty() || value.empty()) return false;
            facts_[key] = {subject, value, false};
            reply = "Got it. I'll remember your " + subject + " is " +
                    value + " during this chat.";
            return true;
        }

        if (std::regex_match(input, match, preference)) {
            const std::string relation = normalized_key(match[1].str());
            const std::string value = trim(match[2].str());
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
        auto words = mikuru::tokenize(input);
        while (!words.empty() &&
               (words.back() == "." || words.back() == "?" ||
                words.back() == "!")) {
            words.pop_back();
        }
        if (words.empty()) return false;

        if (words.size() >= 4 && words[0] == "what" &&
            words[1] == "do" && words[2] == "i") {
            const std::string relation = words[3];
            if ((relation == "like" || relation == "love" ||
                 relation == "prefer") &&
                words.size() == 4) {
                const auto found = facts_.find("i " + relation);
                if (found == facts_.end()) return false;
                reply = "You " + relation + " " + found->second.value + ".";
                return true;
            }
        }

        std::size_t subject_start = 0;
        if (words.size() >= 4 && words[0] == "what" &&
            (words[1] == "is" || words[1] == "s") && words[2] == "my") {
            subject_start = 3;
        } else if (words.size() >= 3 && words[0] == "what's" &&
                   words[1] == "my") {
            subject_start = 2;
        } else {
            return false;
        }
        if (words.back() == "called") words.pop_back();
        if (subject_start >= words.size()) return false;

        std::string subject;
        for (std::size_t i = subject_start; i < words.size(); ++i) {
            if (!subject.empty()) subject.push_back(' ');
            subject += words[i];
        }
        const auto found = facts_.find(subject);
        if (found == facts_.end()) return false;
        reply = found->second.first_person_preference
            ? "You " + found->second.subject + " " + found->second.value + "."
            : "Your " + found->second.subject + " is " +
                  found->second.value + ".";
        return true;
    }

    static bool is_short_statement(const std::string& input) {
        const auto words = mikuru::tokenize(input);
        if (words.empty() || words.size() > 9 ||
            input.find('?') != std::string::npos) {
            return false;
        }
        const auto last = input.find_last_not_of(" \t\r\n");
        if (last == std::string::npos ||
            (input[last] != '.' && input[last] != '!')) {
            return false;
        }
        return words.front() == "i" || words.front() == "my" ||
               words.front() == "we" || words.front() == "it" ||
               words.front() == "that" || words.front() == "today";
    }
};

} // namespace mikuru_conversation

#endif
