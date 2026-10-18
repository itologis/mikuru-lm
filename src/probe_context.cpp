#include "../experiments/mikuru-v3/gru-v1/src/gru_model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace mg = mikuru_gru;
namespace mk = mikuru;
using TokenId = mg::TokenId;

namespace {

struct Scores {
    double music = 0.0;
    double writing = 0.0;
    std::size_t unknown_tokens = 0;
};

void append_text(
    std::vector<TokenId>& prefix,
    const std::string& text,
    const std::unordered_map<std::string, TokenId>& ids
) {
    const auto tokens = mk::encode_tokens(text, ids);
    prefix.insert(prefix.end(), tokens.begin(), tokens.end());
}

std::vector<TokenId> make_prefix(
    const std::string& liked_item,
    bool serialized_history,
    const std::unordered_map<std::string, TokenId>& ids
) {
    std::vector<TokenId> prefix{ids.at("<BOS>"), ids.at("<USER>")};

    const std::string fact =
        "I like " + liked_item + ".";
    const std::string question = "What do I like?";
    const std::string answer_stem = "You like";

    if (serialized_history) {
        // Approximate the chat's multi-turn form:
        // <BOS><USER>fact<ASSISTANT>Okay.<EOS><USER>question<ASSISTANT>answer-stem
        // This is an explicit diagnostic serialization, not a training example.
        append_text(prefix, fact, ids);
        prefix.push_back(ids.at("<ASSISTANT>"));
        append_text(prefix, "Okay.", ids);
        prefix.push_back(ids.at("<EOS>"));
        prefix.push_back(ids.at("<USER>"));
        append_text(prefix, question, ids);
    } else {
        // Training-compatible single-user-turn control:
        // put the remembered fact and follow-up question in one user prompt.
        append_text(prefix, fact + " " + question, ids);
    }

    prefix.push_back(ids.at("<ASSISTANT>"));
    append_text(prefix, answer_stem, ids);
    return prefix;
}

double log_probability(
    const std::vector<double>& logits,
    TokenId target
) {
    if (target >= logits.size() || logits.empty()) {
        return -std::numeric_limits<double>::infinity();
    }

    const double maximum = *std::max_element(logits.begin(), logits.end());
    double sum_exp = 0.0;
    for (const double value : logits) {
        sum_exp += std::exp(value - maximum);
    }
    if (!(sum_exp > 0.0) || !std::isfinite(sum_exp)) {
        return -std::numeric_limits<double>::infinity();
    }
    return logits[target] - maximum - std::log(sum_exp);
}

Scores score_context(
    const mg::Model& model,
    const std::vector<TokenId>& prefix,
    TokenId music_id,
    TokenId writing_id
) {
    std::vector<double> hidden(mg::HIDDEN_DIM, 0.0);
    mg::StepCache cache;

    for (const TokenId token : prefix) {
        cache = mg::forward_step(model, token, hidden);
        hidden = cache.h;
    }

    const auto logits = mg::output_logits(model, cache.h, cache.output_scale);
    Scores result;
    result.music = log_probability(logits, music_id);
    result.writing = log_probability(logits, writing_id);
    for (const TokenId token : prefix) {
        if (token == 1) ++result.unknown_tokens; // <UNK> is vocabulary ID 1.
    }
    return result;
}

bool score_mode(
    const std::string& name,
    bool serialized_history,
    const mg::Model& model,
    const std::unordered_map<std::string, TokenId>& ids,
    TokenId music_id,
    TokenId writing_id
) {
    const Scores a = score_context(
        model, make_prefix("music", serialized_history, ids), music_id, writing_id);
    const Scores b = score_context(
        model, make_prefix("writing", serialized_history, ids), music_id, writing_id);

    const double gap_a = a.music - a.writing;
    const double gap_b = b.music - b.writing;
    const double contrast = gap_a - gap_b;

    std::cout << "\n=== " << name << " ===\n"
              << std::left << std::setw(18) << "History" << std::right
              << std::setw(16) << "logP(music)"
              << std::setw(16) << "logP(writing)"
              << std::setw(18) << "music-writing gap"
              << std::setw(14) << "<UNK> count" << '\n';

    std::cout << std::left << std::setw(18) << "A: likes music"
              << std::right << std::setw(16) << a.music
              << std::setw(16) << a.writing << std::setw(18) << gap_a
              << std::setw(14) << a.unknown_tokens << '\n';
    std::cout << std::left << std::setw(18) << "B: likes writing"
              << std::right << std::setw(16) << b.music
              << std::setw(16) << b.writing << std::setw(18) << gap_b
              << std::setw(14) << b.unknown_tokens << '\n';
    std::cout << "Context contrast (gap A - gap B): " << contrast << '\n';

    if (gap_a > 0.0 && gap_b < 0.0) {
        std::cout << "Result: both contexts rank their stated preference higher.\n";
    } else if (contrast > 0.0) {
        std::cout << "Result: relative scores shift in the expected direction, "
                     "but do not correctly rank both preferences.\n";
    } else if (contrast < 0.0) {
        std::cout << "Result: relative scores shift opposite to the expected direction.\n";
    } else {
        std::cout << "Result: no measured music-vs-writing preference shift.\n";
    }

    return std::isfinite(a.music) && std::isfinite(a.writing) &&
           std::isfinite(b.music) && std::isfinite(b.writing);
}

} // namespace

int main(int argc, char** argv) {
    namespace fs = std::filesystem;
    std::string vocab_path;
    std::string model_path;

    // Match the root Makefile's preferred artifact location.
    const std::string root_vocab = "build/gru/gru-vocab.txt";
    const std::string root_model = "build/gru/gru-model.txt";
    const std::string experiment_vocab =
        "experiments/mikuru-v3/gru-v1/build/gru-vocab.txt";
    const std::string experiment_model =
        "experiments/mikuru-v3/gru-v1/build/gru-model.txt";

    if (fs::exists(root_vocab) && fs::exists(root_model)) {
        vocab_path = root_vocab;
        model_path = root_model;
    } else {
        vocab_path = experiment_vocab;
        model_path = experiment_model;
    }

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "--vocab" || arg == "--model") && i + 1 < argc) {
            const std::string value = argv[++i];
            if (arg == "--vocab") vocab_path = value;
            else model_path = value;
        } else {
            std::cerr << "Usage: probe-context [--vocab PATH] [--model PATH]\n";
            return 2;
        }
    }

    std::vector<std::string> vocab;
    if (!mk::load_vocabulary(vocab_path, vocab)) return 1;
    const auto ids = mk::index_vocabulary(vocab);

    const auto music = ids.find("music");
    const auto writing = ids.find("writing");
    if (music == ids.end() || writing == ids.end() ||
        music->second == ids.at("<UNK>") || writing->second == ids.at("<UNK>")) {
        std::cerr << "Cannot run this probe: both 'music' and 'writing' must be "
                     "distinct, ordinary vocabulary tokens.\n"
                  << "music=" << (music == ids.end() ? "missing" : "present")
                  << ", writing=" << (writing == ids.end() ? "missing" : "present")
                  << "\nNo model files were changed.\n";
        return 3;
    }

    std::mt19937 init_rng(20261019);
    mg::Model model(vocab.size(), init_rng);
    const auto fingerprint = mk::vocabulary_fingerprint(vocab);
    if (!mg::load_model(model_path, model, fingerprint)) return 1;

    std::cout << "Mikuru GRU context probe (read-only)\n"
              << "Vocabulary: " << vocab_path << '\n'
              << "Model:      " << model_path << '\n'
              << "music token ID: " << music->second << '\n'
              << "writing token ID: " << writing->second << '\n'
              << std::fixed << std::setprecision(6);

    const bool single_ok = score_mode(
        "Single-turn control (training-compatible prompt)", false,
        model, ids, music->second, writing->second);
    const bool history_ok = score_mode(
        "Serialized multi-turn history (diagnostic / OOD for V1 training)", true,
        model, ids, music->second, writing->second);

    std::cout << "\nInterpretation: positive contrast means the music-vs-writing score "
                 "shifts in the expected direction across the two contexts.\n"
                 "This is a targeted next-token diagnostic, not proof of "
                 "semantic understanding or natural conversation.\n"
                 "The multi-turn format may be out-of-distribution for a "
                 "checkpoint trained on single-turn dialogue pairs.\n";
    return single_ok && history_ok ? 0 : 1;
}
