#include "conversation_manager.hpp"

#include <iostream>
#include <string>

namespace {

bool check(bool condition, const std::string& description) {
    if (condition) return true;
    std::cerr << "FAIL: " << description << '\n';
    return false;
}

} // namespace

int main() {
    mikuru_conversation::ConversationManager manager;
    std::string reply;
    bool passed = true;

    passed &= check(
        manager.respond("My project is called MoonGarden.", reply) &&
            reply.find("MoonGarden") != std::string::npos,
        "acknowledges and remembers an explicitly stated project"
    );
    passed &= check(
        manager.respond("What is my project called?", reply) &&
            reply == "Your project is MoonGarden.",
        "recalls a fact in a follow-up question"
    );

    passed &= check(
        manager.respond("My project is StarHarbor.", reply) &&
            manager.respond("What's my project?", reply) &&
            reply == "Your project is StarHarbor.",
        "a newly stated value replaces the previous fact"
    );

    passed &= check(
        manager.respond("I like tea.", reply) &&
            reply.find("tea") != std::string::npos &&
            manager.respond("What do I like?", reply) &&
            reply == "You like tea.",
        "remembers and briefly answers a preference"
    );

    passed &= check(
        manager.respond("I won!", reply) &&
            reply == "Got it. Thanks for telling me.",
        "acknowledges a short declarative statement"
    );

    const auto facts_before_unknown = manager.remembered_fact_count();
    passed &= check(
        !manager.respond("How does quantum tunneling work?", reply) &&
            manager.remembered_fact_count() == facts_before_unknown,
        "unknown questions fall through without changing memory"
    );

    manager.reset();
    passed &= check(
        manager.remembered_fact_count() == 0 &&
            !manager.respond("What is my project?", reply),
        "reset clears temporary facts"
    );

    if (!passed) return 1;
    std::cout << "All conversation-manager tests passed.\n";
    return 0;
}
