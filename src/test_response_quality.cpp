#include "response_quality.hpp"

#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

bool check(
    bool condition,
    const std::string& description
) {
    if (condition) return true;
    std::cerr << "FAIL: " << description << '\n';
    return false;
}

} // namespace

int main() {
    using mikuru_response::generated_response_is_usable;
    bool passed = true;

    passed &= check(
        !generated_response_is_usable({}),
        "empty response is rejected"
    );
    for (const auto& token : mikuru::special_tokens()) {
        passed &= check(
            !generated_response_is_usable({token}),
            "special-token leakage is rejected for " + token
        );
    }
    passed &= check(
        !generated_response_is_usable({"The", "the", "."}),
        "case-insensitive adjacent repeated words are rejected"
    );

    passed &= check(
        !generated_response_is_usable(
            {"A", ".", "The", "can", "a", "the", "later", "."}
        ),
        "A. The can a the later. is rejected"
    );
    passed &= check(
        !generated_response_is_usable({"The", "the", ".", "."}),
        "The the.. is rejected"
    );
    passed &= check(
        !generated_response_is_usable({"Can", "the", "you", "the", "."}),
        "Can the you the. is rejected"
    );
    passed &= check(
        !generated_response_is_usable({"Step", "."}),
        "Step. is rejected"
    );

    passed &= check(
        generated_response_is_usable({"Yes", "."}),
        "a legitimate one-word answer is accepted"
    );
    passed &= check(
        generated_response_is_usable({"How", "are", "you", "?"}),
        "a conventional short phrase is accepted"
    );
    passed &= check(
        generated_response_is_usable({"Perhaps", "."}),
        "a short creative-prompt reply is accepted"
    );
    passed &= check(
        generated_response_is_usable({"Blue", "."}),
        "a short factual-prompt reply is accepted"
    );
    passed &= check(
        generated_response_is_usable(
            {"It", "depends", "on", "the", "specific", "context", "."}
        ),
        "a normal medium-length reply is accepted"
    );
    passed &= check(
        generated_response_is_usable(
            {"A", "recurrent", "network", "keeps", "a", "hidden",
             "state", "that", "helps", "it", "process", "sequences", "."}
        ),
        "a normal longer reply is accepted"
    );

    std::mt19937 rng(17);
    const auto creative = mikuru_response::personality_fallback(
        "Write a tiny story about a moonlit garden.", rng
    );
    const auto imagined_creative =
        mikuru_response::personality_fallback(
            "Imagine a tiny festival where stars sing to the ocean.", rng
        );
    const auto technical = mikuru_response::personality_fallback(
        "Help debug this C++ compiler error.", rng
    );
    const auto general = mikuru_response::personality_fallback(
        "What is happening with this unfamiliar topic?", rng
    );
    passed &= check(
        creative.find("idea") != std::string::npos ||
            creative.find("explore") != std::string::npos,
        "creative fallback uses an encouraging, imaginative voice"
    );
    passed &= check(
        imagined_creative.find("idea") != std::string::npos ||
            imagined_creative.find("explore") != std::string::npos,
        "festival and imagination prompts select the creative fallback"
    );
    passed &= check(
        technical.find("example") != std::string::npos ||
            technical.find("code") != std::string::npos,
        "technical fallback offers a practical next step"
    );
    passed &= check(
        general.find("thread") != std::string::npos ||
            general.find("scrambled") != std::string::npos,
        "general fallback is explicit and warm"
    );
    const auto serious = mikuru_response::personality_fallback(
        "I feel worried and exhausted.", rng
    );
    passed &= check(
        serious.find("difficult") != std::string::npos ||
            serious.find("carry") != std::string::npos,
        "serious-topic fallback is gentle"
    );
    const auto rejected_to_fallback =
        mikuru_response::format_or_fallback(
            {"Step", "."},
            "Imagine a story about a festival under the sea.",
            rng
        );
    passed &= check(
        rejected_to_fallback.find("idea") != std::string::npos ||
            rejected_to_fallback.find("explore") != std::string::npos,
        "a rejected generated fragment is replaced by the creative fallback"
    );

    if (!passed) return 1;
    std::cout << "All response-quality tests passed.\n";
    return 0;
}
