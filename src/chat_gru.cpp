#include "../experiments/mikuru-v3/gru-v1/src/gru_model.hpp"

#include "response_quality.hpp"
#include "retrieval.hpp"
#include "system_inspector.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mg = mikuru_gru;
namespace mk = mikuru;
namespace mr = mikuru_response;
using TokenId = mg::TokenId;

namespace {

TokenId sample_top_k(
    const std::vector<std::string>& vocab,
    const std::vector<double>& raw_logits,
    const std::unordered_set<TokenId>& used,
    std::mt19937& rng
) {
    std::vector<std::pair<double, TokenId>> choices;
    choices.reserve(vocab.size());

    for (TokenId i = 0; i < vocab.size(); ++i) {
        if (vocab[i] == "<PAD>" || vocab[i] == "<UNK>" ||
            vocab[i] == "<BOS>" || vocab[i] == "<USER>" ||
            vocab[i] == "<ASSISTANT>") {
            continue;
        }

        double score = raw_logits[i];
        if (used.find(i) != used.end()) score -= 0.25;
        choices.emplace_back(score / 0.9, i);
    }

    if (choices.empty()) return 3;

    const std::size_t k = std::min<std::size_t>(20, choices.size());
    std::partial_sort(
        choices.begin(), choices.begin() +
            static_cast<std::ptrdiff_t>(k), choices.end(),
        [](const auto& a, const auto& b) {
            return a.first > b.first;
        }
    );

    const double maximum = choices.front().first;
    std::vector<double> weights(k);
    for (std::size_t i = 0; i < k; ++i) {
        weights[i] = std::exp(choices[i].first - maximum);
    }

    std::discrete_distribution<std::size_t> pick(
        weights.begin(), weights.end()
    );
    return choices[pick(rng)].second;
}

bool parse_options(
    int argc,
    char** argv,
    std::string& vocab_path,
    std::string& model_path,
    std::uint32_t& seed
) {
    vocab_path = "experiments/mikuru-v3/gru-v1/build/gru-vocab.txt";
    model_path = "experiments/mikuru-v3/gru-v1/build/gru-model.txt";
    seed = std::random_device{}();

    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--vocab" || option == "--model" ||
            option == "--seed") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << option << ".\n";
                return false;
            }
            const std::string value = argv[++i];

            if (option == "--vocab") {
                vocab_path = value;
            } else if (option == "--model") {
                model_path = value;
            } else {
                std::istringstream input(value);
                unsigned long parsed = 0;
                char trailing = '\0';
                if (!(input >> parsed) || (input >> trailing) ||
                    parsed > UINT32_MAX) {
                    std::cerr << "Invalid numeric seed: " << value << '\n';
                    return false;
                }
                seed = static_cast<std::uint32_t>(parsed);
            }
        } else {
            std::cerr
                << "Usage: mikuru-chat-gru [--vocab PATH] [--model PATH] "
                << "[--seed NUMBER]\n";
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    std::string vocab_path;
    std::string model_path;
    std::uint32_t seed = 0;
    if (!parse_options(argc, argv, vocab_path, model_path, seed)) return 2;

    std::vector<std::string> vocab;
    if (!mk::load_vocabulary(vocab_path, vocab)) return 1;

    const auto ids = mk::index_vocabulary(vocab);
    const auto fingerprint = mk::vocabulary_fingerprint(vocab);

    std::mt19937 init_rng(20261019);
    mg::Model model(vocab.size(), init_rng);
    if (!mg::load_model(model_path, model, fingerprint)) return 1;

    for (const auto* marker : {"<BOS>", "<EOS>", "<USER>", "<ASSISTANT>"}) {
        if (ids.find(marker) == ids.end()) {
            std::cerr << "GRU vocabulary is missing required token "
                      << marker << ".\n";
            return 1;
        }
    }

    mikuru_retrieval::Retriever retriever;
    const bool retrieval_ready =
        retriever.load("data/retrieval_dialogue.tsv");
    if (!retrieval_ready) {
        std::cerr
            << "Warning: retrieval unavailable; "
            << "using system answers and GRU generation only.\n";
    }

    const TokenId bos_id = ids.at("<BOS>");
    const TokenId eos_id = ids.at("<EOS>");
    const TokenId user_id = ids.at("<USER>");
    const TokenId assistant_id = ids.at("<ASSISTANT>");
    std::mt19937 rng(seed);

    std::cout
        << "Mikuru hybrid chat | GRU V1 neural fallback\n"
        << "Hi! I'm Mikuru, a local chatbot with a curious, "
        << "virtual-singer sparkle. My replies can be unreliable, "
        << "so let's check important details together.\n"
        << "Architecture: embedding=" << mg::EMBED_DIM
        << ", hidden=" << mg::HIDDEN_DIM << '\n'
        << "Commands: /help, /reset, /quit\n"
        << "Chats stay in memory only and are not saved to disk.\n";

    std::string input;
    while (std::cout << "\nYou> " && std::getline(std::cin, input)) {
        if (input == "/quit" || input == "/exit") break;
        if (input == "/help") {
            std::cout
                << "I'll check known answers first, then use my small GRU "
                << "language model. Its replies can be off-key, so verify "
                << "important details. /reset starts fresh; /quit exits.\n";
            continue;
        }
        if (input == "/reset") {
            std::cout << "Mikuru> Fresh prompt ready!\n";
            continue;
        }
        if (mk::tokenize(input).empty()) {
            std::cout << "Mikuru> Send me a few words to work with.\n";
            continue;
        }

        std::string system_answer;
        if (mikuru_system::answer_system_question(input, system_answer)) {
            std::cout << "Mikuru> " << system_answer;
            if (system_answer.empty() || system_answer.back() != '\n') {
                std::cout << '\n';
            }
            continue;
        }

        if (retrieval_ready) {
            const auto match = retriever.find(input);
            if (match.index >= 0) {
                const auto& item = retriever.dialogue(match.index);
                std::cout << "Mikuru> " << item.reply << '\n';
                continue;
            }
        }

        std::vector<TokenId> prefix{bos_id, user_id};
        const auto prompt_ids = mk::encode_tokens(input, ids);
        prefix.insert(prefix.end(), prompt_ids.begin(), prompt_ids.end());
        prefix.push_back(assistant_id);

        std::vector<double> hidden(mg::HIDDEN_DIM, 0.0);
        mg::StepCache cache;
        for (const TokenId token : prefix) {
            cache = mg::forward_step(model, token, hidden);
            hidden = cache.h;
        }

        std::vector<std::string> response;
        std::unordered_set<TokenId> used;
        constexpr std::size_t MAX_NEW_TOKENS = 40;

        for (std::size_t step = 0; step < MAX_NEW_TOKENS; ++step) {
            const auto logits = mg::output_logits(
                model, cache.h, cache.output_scale
            );
            const TokenId next =
                sample_top_k(vocab, logits, used, rng);
            if (next == eos_id || next >= vocab.size()) break;

            response.push_back(vocab[next]);
            used.insert(next);
            cache = mg::forward_step(model, next, hidden);
            hidden = cache.h;
        }

        std::cout << "Mikuru> "
                  << mr::format_or_fallback(response, input, rng) << '\n';
    }

    std::cout << "Mikuru> See you next time.\n";
    return 0;
}
