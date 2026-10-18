#include "gru_model.hpp"

#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mg = mikuru_gru;
namespace mk = mikuru;
using TokenId = mg::TokenId;

static TokenId sample_top_k(
    const mg::Model& model,
    const std::vector<std::string>& vocab,
    const std::vector<double>& raw_logits,
    const std::unordered_set<TokenId>& used,
    std::mt19937& rng
) {
    std::vector<std::pair<double, TokenId>> choices;
    choices.reserve(vocab.size());

    for (TokenId i = 0; i < vocab.size(); ++i) {
        double score = raw_logits[i];

        if (vocab[i] == "<PAD>" || vocab[i] == "<UNK>" ||
            vocab[i] == "<BOS>" || vocab[i] == "<USER>" ||
            vocab[i] == "<ASSISTANT>") {
            continue;
        }
        if (used.find(i) != used.end()) score -= 0.25;
        score /= 0.9;
        choices.emplace_back(score, i);
    }

    if (choices.empty()) return 3; // <EOS>

    const std::size_t k = std::min<std::size_t>(20, choices.size());
    std::partial_sort(
        choices.begin(), choices.begin() + static_cast<std::ptrdiff_t>(k),
        choices.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );

    const double maximum = choices.front().first;
    std::vector<double> weights(k);
    for (std::size_t i = 0; i < k; ++i)
        weights[i] = std::exp(choices[i].first - maximum);

    std::discrete_distribution<std::size_t> pick(weights.begin(), weights.end());
    return choices[pick(rng)].second;
}

int main() {
    const std::string vocab_path = "build/gru-vocab.txt";
    const std::string model_path = "build/gru-model.txt";

    std::vector<std::string> vocab;
    if (!mk::load_vocabulary(vocab_path, vocab)) return 1;
    const auto ids = mk::index_vocabulary(vocab);
    const auto hash = mk::vocabulary_fingerprint(vocab);

    std::mt19937 init_rng(20261019);
    mg::Model model(vocab.size(), init_rng);
    if (!mg::load_model(model_path, model, hash)) return 1;

    std::mt19937 rng(std::random_device{}());
    std::cout << "Mikuru GRU V1 | standalone neural experiment\n"
              << "Embedding: 32 | GRU hidden: 64\n"
              << "Commands: /help, /reset, /quit\n"
              << "Chats stay in memory only and are not saved to disk.\n";

    std::string input;
    while (std::cout << "\nYou> " && std::getline(std::cin, input)) {
        if (input == "/quit" || input == "/exit") break;
        if (input == "/help") {
            std::cout << "Standalone GRU generator. /reset starts a fresh prompt; /quit exits.\n";
            continue;
        }
        if (input == "/reset") {
            std::cout << "Mikuru> Fresh prompt ready.\n";
            continue;
        }

        const auto tokens = mk::tokenize(input);
        if (tokens.empty()) {
            std::cout << "Mikuru> Give me a few words to work with.\n";
            continue;
        }

        std::vector<TokenId> prefix = {
            ids.at("<BOS>"), ids.at("<USER>")
        };
        const auto prompt_ids = mk::encode_tokens(input, ids);
        prefix.insert(prefix.end(), prompt_ids.begin(), prompt_ids.end());
        prefix.push_back(ids.at("<ASSISTANT>"));

        std::vector<double> hidden(mg::HIDDEN_DIM, 0.0);
        mg::StepCache cache;
        for (const auto id : prefix) {
            cache = mg::forward_step(model, id, hidden);
            hidden = cache.h;
        }

        std::vector<std::string> response;
        std::unordered_set<TokenId> used;
        constexpr std::size_t MAX_NEW_TOKENS = 40;

        for (std::size_t step = 0; step < MAX_NEW_TOKENS; ++step) {
            const auto logits = mg::output_logits(
                model, cache.h, cache.output_scale);
            const TokenId next = sample_top_k(model, vocab, logits, used, rng);

            if (next == ids.at("<EOS>")) break;
            if (next >= vocab.size()) break;

            response.push_back(vocab[next]);
            used.insert(next);

            cache = mg::forward_step(model, next, hidden);
            hidden = cache.h;
        }

        if (response.empty()) {
            std::cout << "Mikuru> I couldn't form a response this time.\n";
        } else {
            std::cout << "Mikuru> " << mk::format_text(response) << '\n';
        }
    }

    std::cout << "Mikuru> See you next time.\n";
    return 0;
}
