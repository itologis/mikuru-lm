#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mikuru_retrieval {

struct Dialogue {
    std::string prompt;
    std::string reply;
    std::string normalized;
    std::map<std::string, double> features;
    double norm = 0.0;
};

struct Match {
    int index = -1;
    double score = 0.0;
    bool exact = false;
    double query_coverage = 0.0;
};

const std::set<std::string> STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "so", "to",
    "of", "in", "on", "at", "for", "from", "with", "by", "about",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had",
    "i", "me", "my", "mine", "you", "your", "yours",
    "we", "our", "ours", "they", "them", "their", "it", "its",
    "what", "when", "where", "why", "how", "who", "which",
    "can", "could", "would", "should", "will",
    "please", "just", "really", "very",
    "explain", "explains", "explaining", "simple", "simply", "easily",
    "beginner", "friendly", "language", "tell", "show", "describe",
    "help", "write", "writing", "make", "create", "want", "need",
    "difference", "between", "versus", "vs", "work", "works", "working",
    "this", "that", "myself"
};

inline bool ascii_alnum(unsigned char c) {
    return c < 128 && std::isalnum(c) != 0;
}

inline std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> result;
    std::size_t i = 0;

    while (i < text.size()) {
        const unsigned char c = static_cast<unsigned char>(text[i]);

        if (ascii_alnum(c)) {
            std::string word;

            while (i < text.size()) {
                const unsigned char current =
                    static_cast<unsigned char>(text[i]);

                if (ascii_alnum(current)) {
                    word.push_back(
                        static_cast<char>(std::tolower(current))
                    );
                    ++i;
                } else if (
                    text[i] == '\'' && i + 1 < text.size() &&
                    ascii_alnum(
                        static_cast<unsigned char>(text[i + 1])
                    )
                ) {
                    word.push_back('\'');
                    ++i;
                } else {
                    break;
                }
            }

            result.push_back(word);
            continue;
        }

        // Preserve UTF-8 code points as individual tokens.
        // This is lightweight and dependency-free; it is not
        // a full Japanese word segmenter or emoji grapheme tokenizer.
        if (c >= 128) {
            std::size_t length = 1;
            if ((c & 0xE0) == 0xC0) length = 2;
            else if ((c & 0xF0) == 0xE0) length = 3;
            else if ((c & 0xF8) == 0xF0) length = 4;

            length = std::min(length, text.size() - i);
            result.push_back(text.substr(i, length));
            i += length;
        } else {
            ++i;
        }
    }

    return result;
}

inline std::string normalize(const std::string& text) {
    const auto parts = tokenize(text);
    std::string result;

    for (const auto& part : parts) {
        if (!result.empty()) result.push_back(' ');
        result += part;
    }

    return result;
}

inline std::vector<std::string> content_tokens(const std::string& text) {
    std::vector<std::string> result;

    for (const auto& token : tokenize(text)) {
        if (STOPWORDS.find(token) == STOPWORDS.end()) {
            result.push_back(token);
        }
    }

    return result;
}

enum class Intent {
    General, Explanation, CreativeWriting, Coding, Troubleshooting, Comparison
};

inline bool has_any_token(
    const std::string& text,
    std::initializer_list<std::string> choices
) {
    for (const auto& word : tokenize(text))
        if (std::find(choices.begin(), choices.end(), word) != choices.end())
            return true;
    return false;
}

inline Intent classify_intent(const std::string& text) {
    if (has_any_token(text, {"difference", "compare", "comparison", "versus", "vs"}))
        return Intent::Comparison;
    if (has_any_token(text, {"story", "stories", "novel", "fiction", "plot", "poem", "screenplay"}))
        return Intent::CreativeWriting;
    if (has_any_token(text, {"error", "bug", "crash", "crashed", "debug", "debugging", "broken", "fix", "fixing", "issue"}))
        return Intent::Troubleshooting;
    if (has_any_token(text, {"program", "programming", "code", "coding", "software", "script", "compiler", "interpreter", "function", "python", "javascript", "java", "sql", "html", "css"}))
        return Intent::Coding;
    if (has_any_token(text, {"explain", "explanation", "define", "definition", "meaning", "describe"}))
        return Intent::Explanation;
    return Intent::General;
}

inline bool intent_conflict(const std::string& query, const std::string& candidate) {
    const Intent a = classify_intent(query);
    const Intent b = classify_intent(candidate);
    return a != Intent::General && b != Intent::General && a != b;
}

inline int sentiment_polarity(const std::string& text) {
    const bool positive = has_any_token(text, {
        "passed", "pass", "proud", "happy", "glad", "excited",
        "celebrate", "celebration", "succeeded", "success", "won",
        "fixed", "solved", "finished", "completed", "relieved",
        "great", "good"
    });
    const bool negative = has_any_token(text, {
        "failed", "fail", "disappointed", "exhausted", "tired", "rough",
        "sad", "upset", "stressed", "stressful", "anxious", "worried",
        "lonely", "frustrated", "angry", "terrible", "awful",
        "overwhelmed", "scared", "miserable", "bad", "nervous"
    });
    if (positive == negative) return 0;
    return positive ? 1 : -1;
}

inline bool sentiment_conflict(const std::string& query, const std::string& candidate) {
    const int a = sentiment_polarity(query);
    const int b = sentiment_polarity(candidate);
    return a != 0 && b != 0 && a != b;
}

inline double topic_overlap(const std::string& query, const std::string& candidate) {
    const auto q = content_tokens(query);
    const auto c = content_tokens(candidate);
    const std::set<std::string> qt(q.begin(), q.end());
    const std::set<std::string> ct(c.begin(), c.end());
    if (qt.empty()) return 0.0;
    std::size_t shared = 0;
    for (const auto& word : qt)
        if (ct.find(word) != ct.end()) ++shared;
    return static_cast<double>(shared) / static_cast<double>(qt.size());
}

constexpr double MIN_SIMILARITY = 0.50;
constexpr double MIN_TOPIC_OVERLAP = 0.50;

inline bool load_dialogues(
    const std::string& path,
    std::vector<Dialogue>& dialogues
) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Cannot open dialogue file: " << path << '\n';
        return false;
    }

    std::string line;
    if (!std::getline(in, line) || line != "user\tassistant") {
        std::cerr << "Invalid TSV header.\n";
        return false;
    }

    std::size_t line_number = 1;

    while (std::getline(in, line)) {
        ++line_number;
        if (line.empty()) continue;

        const std::size_t tab = line.find('\t');
        if (tab == std::string::npos) {
            std::cerr << "Missing tab at line " << line_number << '\n';
            return false;
        }

        Dialogue item;
        item.prompt = line.substr(0, tab);
        item.reply = line.substr(tab + 1);

        if (item.prompt.empty() || item.reply.empty()) {
            std::cerr << "Empty prompt or reply at line "
                      << line_number << '\n';
            return false;
        }

        item.normalized = normalize(item.prompt);
        dialogues.push_back(std::move(item));
    }

    if (dialogues.empty()) {
        std::cerr << "No dialogue pairs loaded.\n";
        return false;
    }

    return true;
}

inline std::map<std::string, std::size_t> document_frequency(
    const std::vector<Dialogue>& dialogues
) {
    std::map<std::string, std::size_t> df;

    for (const auto& dialogue : dialogues) {
        const auto tokens = content_tokens(dialogue.prompt);
        const std::set<std::string> unique(tokens.begin(), tokens.end());

        for (const auto& word : unique) ++df[word];
    }

    return df;
}

inline std::map<std::string, double> make_features(
    const std::string& text,
    const std::map<std::string, std::size_t>& df,
    std::size_t document_count
) {
    const auto tokens = content_tokens(text);
    std::map<std::string, std::size_t> frequencies;

    for (const auto& token : tokens) {
        if (df.find(token) != df.end()) ++frequencies[token];
    }

    std::map<std::string, double> features;

    for (const auto& entry : frequencies) {
        const double frequency =
            1.0 + std::log(static_cast<double>(entry.second));

        const double idf = std::log(
            (static_cast<double>(document_count) + 1.0) /
            (static_cast<double>(df.at(entry.first)) + 1.0)
        ) + 1.0;

        features[entry.first] = frequency * idf;
    }

    return features;
}

inline double vector_norm(const std::map<std::string, double>& features) {
    double sum = 0.0;

    for (const auto& item : features) {
        sum += item.second * item.second;
    }

    return std::sqrt(sum);
}

inline double cosine_similarity(
    const std::map<std::string, double>& a,
    const std::map<std::string, double>& b,
    double norm_a,
    double norm_b
) {
    if (norm_a == 0.0 || norm_b == 0.0) return 0.0;

    double dot = 0.0;

    for (const auto& item : a) {
        const auto found = b.find(item.first);
        if (found != b.end()) {
            dot += item.second * found->second;
        }
    }

    return dot / (norm_a * norm_b);
}

class Retriever {
public:
    bool load(const std::string& path) {
        if (!load_dialogues(path, dialogues_)) return false;

        df_ = document_frequency(dialogues_);

        for (auto& dialogue : dialogues_) {
            dialogue.features = make_features(
                dialogue.prompt, df_, dialogues_.size()
            );
            dialogue.norm = vector_norm(dialogue.features);
        }

        return true;
    }

    std::size_t size() const {
        return dialogues_.size();
    }

    Match find(const std::string& query) const {
        Match best;

        if (normalize(query).empty()) return best;

        // Exact normalized prompts take priority over fuzzy matches.
        const std::string normalized_query = normalize(query);

        for (std::size_t i = 0; i < dialogues_.size(); ++i) {
            if (dialogues_[i].normalized == normalized_query) {
                best.index = static_cast<int>(i);
                best.score = 1.0;
                best.exact = true;
                best.query_coverage = 1.0;
                return best;
            }
        }

        const auto raw_content = content_tokens(query);
        if (raw_content.empty()) return best;

        std::size_t known_tokens = 0;
        for (const auto& token : raw_content)
            if (df_.find(token) != df_.end()) ++known_tokens;
        if (known_tokens == 0) return best;

        const auto query_features =
            make_features(query, df_, dialogues_.size());
        const double query_norm = vector_norm(query_features);
        if (query_norm == 0.0) return best;

        double top_score = -1.0;
        double top_overlap = 0.0;
        int top_index = -1;

        for (std::size_t i = 0; i < dialogues_.size(); ++i) {
            const Dialogue& candidate = dialogues_[i];
            const double score = cosine_similarity(
                query_features, candidate.features,
                query_norm, candidate.norm
            );
            if (score < MIN_SIMILARITY) continue;

            const double overlap = topic_overlap(query, candidate.prompt);
            if (overlap < MIN_TOPIC_OVERLAP) continue;
            if (intent_conflict(query, candidate.prompt)) continue;
            if (sentiment_conflict(query, candidate.prompt)) continue;

            if (score > top_score) {
                top_score = score;
                top_overlap = overlap;
                top_index = static_cast<int>(i);
            }
        }

        if (top_index < 0) return best;
        best.index = top_index;
        best.score = top_score;
        best.exact = false;
        best.query_coverage = top_overlap;
        return best;
    }

    const Dialogue& dialogue(int index) const {
        return dialogues_.at(static_cast<std::size_t>(index));
    }

private:
    std::vector<Dialogue> dialogues_;
    std::map<std::string, std::size_t> df_;
};

const std::string FALLBACK =
    "I don't have a close example for that yet. "
    "Could you tell me a little more about what you mean?";

inline void answer(Retriever& retriever, const std::string& query) {
    const Match match = retriever.find(query);

    if (match.index < 0) {
        std::cout << "Mikuru> " << FALLBACK << '\n';
        std::cout << "Match: none\n";
        return;
    }

    const Dialogue& item = retriever.dialogue(match.index);

    std::cout << "Mikuru> " << item.reply << '\n';
    std::cout << "Matched prompt: " << item.prompt << '\n';

    if (match.exact) {
        std::cout << "Match type: exact after normalization\n";
    } else {
        std::cout << "Match type: lexical similarity\n";
        std::cout << "Similarity score: "
                  << std::fixed << std::setprecision(3)
                  << match.score << '\n';
        std::cout << "Query topic overlap: "
                  << std::fixed << std::setprecision(1)
                  << match.query_coverage * 100.0 << "%\n";
    }
}

inline void benchmark(Retriever& retriever) {
    const std::vector<std::string> prompts = {
        "Hey, Mikuru",
        "I want to learn drawing",
        "I'm having a rough day",
        "I don't understand this code",
        "What is a token?",
        "My program crashed",
        "Tell me something fun",
        "Can you explain a neural network simply?",
        "How can I improve this dialogue model?",
        "I made a mistake in front of people"
    };

    for (std::size_t i = 0; i < prompts.size(); ++i) {
        std::cout << "\n[" << (i + 1) << "] You> "
                  << prompts[i] << '\n';
        answer(retriever, prompts[i]);
    }
}

} // namespace mikuru_retrieval
