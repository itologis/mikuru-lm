
#include "dialogue_common.hpp"
#include "retrieval.hpp"
#include "response_quality.hpp"
#include "system_inspector.hpp"

namespace {
using namespace mikuru;

TokenId sample_next(
    std::vector<double> probabilities,
    std::mt19937& rng,
    const std::vector<std::string>& vocab
) {
    for (TokenId i = 0; i < vocab.size(); ++i) {
        const auto& token = vocab[i];

        if (token == "<PAD>" || token == "<UNK>" ||
            token == "<BOS>" || token == "<USER>" ||
            token == "<ASSISTANT>") {
            probabilities[i] = 0.0;
        }
    }

    constexpr double temperature = 0.85;

    std::vector<TokenId> order(probabilities.size());
    std::iota(order.begin(), order.end(), 0);

    const std::size_t k = std::min<std::size_t>(12, order.size());

    std::partial_sort(
        order.begin(),
        order.begin() + static_cast<std::ptrdiff_t>(k),
        order.end(),
        [&probabilities](TokenId a, TokenId b) {
            return probabilities[a] > probabilities[b];
        }
    );

    std::vector<double> weights(probabilities.size(), 0.0);

    for (std::size_t i = 0; i < k; ++i) {
        const TokenId id = order[i];

        if (probabilities[id] > 0.0) {
            weights[id] = std::pow(
                probabilities[id], 1.0 / temperature
            );
        }
    }

    const double total = std::accumulate(
        weights.begin(), weights.end(), 0.0
    );

    if (!(total > 0.0)) return 3; // <EOS>

    std::discrete_distribution<TokenId> distribution(
        weights.begin(), weights.end()
    );

    return distribution(rng);
}

} // namespace

int main() {
    using namespace mikuru;

    const std::string vocab_path = "build/dialogue-vocab.txt";
    const std::string model_path = "build/mikuru-dialogue.weights";

    std::vector<std::string> vocab;

    if (!load_vocabulary(vocab_path, vocab)) return 1;

    const auto ids = index_vocabulary(vocab);
    const auto fingerprint = vocabulary_fingerprint(vocab);

    Network network(vocab.size());

    if (!load_model(model_path, network, fingerprint)) return 1;

    // Preserve the existing retrieval-first design.
    mikuru_retrieval::Retriever retriever;

    const bool retrieval_ready =
        retriever.load("data/retrieval_dialogue.tsv");

    if (!retrieval_ready) {
        std::cerr
            << "Warning: retrieval unavailable; "
            << "using neural generation only.\n";
    }

    const TokenId pad_id = ids.at("<PAD>");
    const TokenId bos_id = ids.at("<BOS>");
    const TokenId eos_id = ids.at("<EOS>");
    const TokenId user_id = ids.at("<USER>");
    const TokenId assistant_id = ids.at("<ASSISTANT>");

    std::mt19937 rng(std::random_device{}());
    std::vector<TokenId> history{bos_id};

    std::cout
        << "Mikuru v0.2 prompt-conditioned chat\n"
        << "Hybrid chat: retrieval first, neural generation fallback.\n"
        << "Commands: /help, /reset, /quit\n"
        << "Chats stay in memory only and are not saved to disk.\n";

    std::string input;

    while (true) {
        std::cout << "\nYou> " << std::flush;

        if (!std::getline(std::cin, input)) break;

        if (input == "/quit" || input == "/exit") break;

        if (input == "/help") {
            std::cout
                << "I check for a relevant example first, then use "
                << "neural generation if needed. "
                << "/reset clears context; /quit exits.\n";
            continue;
        }

        if (input == "/reset") {
            history.assign(1, bos_id);
            std::cout << "Mikuru> Context cleared. Fresh start!\n";
            continue;
        }

        if (tokenize(input).empty()) {
            std::cout << "Mikuru> Send me a few words and let's try.\n";
            continue;
        }

        // System questions retain their existing priority.
        std::string system_answer;

        if (mikuru_system::answer_system_question(
                input, system_answer
            )) {
            std::cout << "Mikuru> " << system_answer;

            if (system_answer.empty() ||
                system_answer.back() != '\n') {
                std::cout << '\n';
            }

            continue;
        }

        // Check retrieval before attempting neural generation.
        if (retrieval_ready) {
            const auto match = retriever.find(input);

            if (match.index >= 0) {
                const auto& item =
                    retriever.dialogue(match.index);

                std::cout << "Mikuru> " << item.reply << '\n';
                continue;
            }
        }

        // Neural fallback starts with a fresh prompt context.
        history.assign(1, bos_id);
        history.push_back(user_id);

        const auto prompt_ids = encode_tokens(input, ids);

        history.insert(
            history.end(),
            prompt_ids.begin(),
            prompt_ids.end()
        );

        history.push_back(assistant_id);

        std::vector<std::string> response;

        constexpr int MAX_NEW_TOKENS = 28;

        for (int step = 0; step < MAX_NEW_TOKENS; ++step) {
            std::array<TokenId, CONTEXT_SIZE> context;
            context.fill(pad_id);

            const std::size_t available =
                std::min(CONTEXT_SIZE, history.size());

            for (std::size_t i = 0; i < available; ++i) {
                context[CONTEXT_SIZE - available + i] =
                    history[history.size() - available + i];
            }

            Sample sample;
            sample.context = context;
            sample.prompt = prompt_ids;
            sample.target = 0;

            auto probabilities =
                forward(sample, network).probabilities;

            const TokenId next = sample_next(
                std::move(probabilities), rng, vocab
            );

            if (next == eos_id) break;
            if (next >= vocab.size()) break;

            response.push_back(vocab[next]);
            history.push_back(next);
        }

        history.push_back(eos_id);

        // Reject some obvious failures before displaying generation.
        std::cout << "Mikuru> "
                  << mikuru_response::format_or_fallback(
                         response, input, rng
                     )
                  << '\n';

        constexpr std::size_t MAX_HISTORY = 256;

        if (history.size() > MAX_HISTORY) {
            std::vector<TokenId> recent(
                history.end() -
                    static_cast<std::ptrdiff_t>(MAX_HISTORY - 1),
                history.end()
            );

            history.clear();
            history.push_back(bos_id);

            history.insert(
                history.end(),
                recent.begin(),
                recent.end()
            );
        }
    }

    std::cout << "Mikuru> See you next time.\n";

    return 0;
}
