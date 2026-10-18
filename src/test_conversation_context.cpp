#include "conversation_context.hpp"
#include "../experiments/mikuru-v3/gru-v1/src/gru_model.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

bool check(bool condition, const std::string& description) {
    if (condition) return true;
    std::cerr << "FAIL: " << description << '\n';
    return false;
}

} // namespace

int main() {
    using mikuru_context::ConversationContext;
    bool passed = true;
    ConversationContext context;

    const std::vector<std::size_t> previous_user{11, 12};
    const std::vector<std::size_t> previous_reply{21, 22};
    context.remember_turn(4, previous_user, 5, previous_reply, 3);

    const auto follow_up = context.build_prefix(2, 4, {31}, 5);
    const std::vector<std::size_t> expected{
        2, 4, 11, 12, 5, 21, 22, 3, 4, 31, 5
    };
    passed &= check(
        follow_up == expected,
        "follow-up prefix preserves both messages, markers, and turn order"
    );

    std::mt19937 model_rng(97);
    mikuru_gru::Model model(256, model_rng);
    const auto final_hidden = [&model](
        const std::vector<std::size_t>& prefix
    ) {
        std::vector<double> hidden(mikuru_gru::HIDDEN_DIM, 0.0);
        for (const auto token : prefix) {
            hidden = mikuru_gru::forward_step(
                model, token, hidden
            ).h;
        }
        return hidden;
    };
    const auto with_history = final_hidden(follow_up);
    const ConversationContext empty_context;
    const auto without_history = final_hidden(
        empty_context.build_prefix(2, 4, {31}, 5)
    );
    bool state_changed = false;
    for (std::size_t i = 0; i < with_history.size(); ++i) {
        if (std::abs(with_history[i] - without_history[i]) > 1e-12) {
            state_changed = true;
            break;
        }
    }
    passed &= check(
        state_changed,
        "earlier-turn tokens alter the recurrent state used for generation"
    );

    ConversationContext budget_context;
    const std::vector<std::size_t> fifty_tokens(50, 7);
    for (std::size_t i = 0; i < 3; ++i) {
        budget_context.remember_turn(
            4, fifty_tokens, 5, fifty_tokens, 3
        );
    }
    const std::vector<std::size_t> full_prompt(
        ConversationContext::MAX_CURRENT_PROMPT_TOKENS, 8
    );
    const auto budgeted = budget_context.build_prefix(
        2, 4, full_prompt, 5
    );
    passed &= check(
        budgeted.size() <= ConversationContext::MAX_CONTEXT_TOKENS &&
            budgeted.size() == 234,
        "context budget retains the newest complete exchange without partial turns"
    );

    context.reset();
    passed &= check(
        context.retained_turns() == 0 &&
            context.build_prefix(2, 4, {31}, 5) ==
                std::vector<std::size_t>{2, 4, 31, 5},
        "reset clears all previous turns from the next prefix"
    );

    for (std::size_t i = 0; i < ConversationContext::MAX_TURNS + 2; ++i) {
        context.remember_turn(4, {100 + i}, 5, {200 + i}, 3);
    }
    passed &= check(
        context.retained_turns() == ConversationContext::MAX_TURNS,
        "in-memory history keeps no more than the configured turn count"
    );

    std::vector<std::size_t> long_prompt(
        ConversationContext::MAX_CURRENT_PROMPT_TOKENS, 7
    );
    const auto bounded = context.build_prefix(2, 4, long_prompt, 5);
    passed &= check(
        bounded.size() <= ConversationContext::MAX_CONTEXT_TOKENS,
        "serialized prefix stays within the token budget"
    );

    std::vector<std::size_t> too_long_prompt(
        ConversationContext::MAX_CONTEXT_TOKENS, 7
    );
    passed &= check(
        context.build_prefix(2, 4, too_long_prompt, 5).empty(),
        "prefix construction rejects an oversized current prompt"
    );

    if (!passed) return 1;
    std::cout << "All conversation-context tests passed.\n";
    return 0;
}
