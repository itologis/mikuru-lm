#ifndef MIKURU_CONVERSATION_CONTEXT_HPP
#define MIKURU_CONVERSATION_CONTEXT_HPP

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

namespace mikuru_context {

class ConversationContext {
public:
    static constexpr std::size_t MAX_TURNS = 6;
    static constexpr std::size_t MAX_CONTEXT_TOKENS = 256;
    static constexpr std::size_t MAX_CURRENT_PROMPT_TOKENS = 128;

    using TokenId = std::size_t;

    std::vector<TokenId> build_prefix(
        TokenId bos,
        TokenId user_marker,
        const std::vector<TokenId>& prompt,
        TokenId assistant_marker
    ) const {
        std::vector<TokenId> prefix{bos};
        const std::size_t current_size = prompt.size() + 3;
        if (current_size > MAX_CONTEXT_TOKENS) return {};

        const std::size_t history_budget =
            MAX_CONTEXT_TOKENS - current_size;
        std::size_t selected_tokens = 0;
        std::size_t first_turn = turns_.size();

        while (first_turn > 0 && turns_.size() - first_turn < MAX_TURNS) {
            const std::size_t turn_size = turns_[first_turn - 1].size();
            if (turn_size > history_budget - selected_tokens) break;
            selected_tokens += turn_size;
            --first_turn;
        }

        for (std::size_t i = first_turn; i < turns_.size(); ++i) {
            prefix.insert(prefix.end(), turns_[i].begin(), turns_[i].end());
        }
        prefix.push_back(user_marker);
        prefix.insert(prefix.end(), prompt.begin(), prompt.end());
        prefix.push_back(assistant_marker);
        return prefix;
    }

    void remember_turn(
        TokenId user_marker,
        const std::vector<TokenId>& user_tokens,
        TokenId assistant_marker,
        const std::vector<TokenId>& assistant_tokens,
        TokenId eos
    ) {
        std::vector<TokenId> turn;
        turn.reserve(user_tokens.size() + assistant_tokens.size() + 3);
        turn.push_back(user_marker);
        turn.insert(turn.end(), user_tokens.begin(), user_tokens.end());
        turn.push_back(assistant_marker);
        turn.insert(turn.end(), assistant_tokens.begin(),
                    assistant_tokens.end());
        turn.push_back(eos);
        turns_.push_back(std::move(turn));

        if (turns_.size() > MAX_TURNS) {
            turns_.erase(turns_.begin());
        }
    }

    void reset() {
        turns_.clear();
    }

    std::size_t retained_turns() const {
        return turns_.size();
    }

private:
    std::vector<std::vector<TokenId>> turns_;
};

} // namespace mikuru_context

#endif
