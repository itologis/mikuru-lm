#include "intent_router.hpp"

#include <iostream>
#include <string>
#include <vector>

struct TestCase {
    std::string input;
    mikuru_system::Intent expected;
};

const char* intent_name(mikuru_system::Intent intent) {
    using mikuru_system::Intent;

    switch (intent) {
        case Intent::None: return "None";
        case Intent::Disk: return "Disk";
        case Intent::Memory: return "Memory";
        case Intent::DiskAndMemory: return "DiskAndMemory";
    }

    return "Unknown";
}

int main() {
    using mikuru_system::Intent;

    const std::vector<TestCase> tests = {
        {"How much disk space do I have?", Intent::Disk},
        {"Is my storage almost full?", Intent::Disk},
        {"How much space is left on my drive?", Intent::Disk},
        {"Can you check RAM?", Intent::Memory},
        {"How much RAM is available?", Intent::Memory},
        {"How much memory is available?", Intent::Memory},
        {"Is my laptop running out of memory?", Intent::Memory},
        {"Show me disk space and memory.", Intent::DiskAndMemory},
        {"What's my RAM usage?", Intent::Memory},

        {"My program crashed.", Intent::None},
        {"Can you check its storage requirements?", Intent::None},
        {"Explain storage requirements.", Intent::None},
        {"Explain memory management.", Intent::None},
        {"I have a good memory.", Intent::None},
        {"Can you explain how RAM works?", Intent::None},
        {"What does disk space mean in game design?", Intent::None}
    };

    int failures = 0;

    for (const auto& test : tests) {
        const auto actual = mikuru_system::classify(test.input);
        const bool passed = actual == test.expected;

        std::cout << (passed ? "PASS: " : "FAIL: ")
                  << test.input;

        if (!passed) {
            std::cout << " [expected " << intent_name(test.expected)
                      << ", got " << intent_name(actual) << "]";
            ++failures;
        }

        std::cout << '\n';
    }

    std::cout << '\n' << tests.size() - failures
              << '/' << tests.size() << " tests passed.\n";

    return failures == 0 ? 0 : 1;
}
