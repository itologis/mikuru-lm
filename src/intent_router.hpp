#ifndef MIKURU_INTENT_ROUTER_HPP
#define MIKURU_INTENT_ROUTER_HPP

#include <cctype>
#include <initializer_list>
#include <string>

namespace mikuru_system {

enum class Intent {
    None,
    Disk,
    Memory,
    DiskAndMemory
};

inline std::string normalize(const std::string& input) {
    std::string result;
    bool previous_space = true;

    for (std::size_t i = 0; i < input.size(); ++i) {
        // Treat straight and curly apostrophes as equivalent.
        if (input[i] == '\'') {
            continue;
        }

        if (input.compare(i, 3, "\xE2\x80\x99") == 0) {
            i += 2;
            continue;
        }

        const auto c = static_cast<unsigned char>(input[i]);

        if (std::isalnum(c)) {
            result += static_cast<char>(std::tolower(c));
            previous_space = false;
        } else if (!previous_space) {
            result += ' ';
            previous_space = true;
        }
    }

    if (!result.empty() && result.back() == ' ') {
        result.pop_back();
    }

    return result;
}

inline bool has_phrase(
    const std::string& normalized_text,
    std::initializer_list<const char*> phrases) {

    const std::string padded_text = " " + normalized_text + " ";

    for (const char* phrase : phrases) {
        const std::string normalized_phrase = normalize(phrase);
        const std::string target = " " + normalized_phrase + " ";

        if (padded_text.find(target) != std::string::npos) {
            return true;
        }
    }

    return false;
}

inline Intent classify(const std::string& input) {
    const std::string text = normalize(input);

    // Combined checks must be explicit. Mentioning two topics
    // without asking for a system check should not trigger a tool.
    const bool both = has_phrase(text, {
        "show me disk space and memory",
        "show me disk space and ram",
        "check disk space and memory",
        "check disk space and ram",
        "disk space and memory",
        "disk space and ram"
    });

    const bool disk = has_phrase(text, {
        "how much disk space do i have",
        "how much disk space is available",
        "how much storage do i have",
        "how much space do i have",
        "how much space is left on my drive",
        "how much space is left on my disk",
        "space left on my drive",
        "space left on my disk",
        "free space on my drive",
        "free space on my disk",
        "available disk space",
        "free disk space",
        "check disk space",
        "check my disk",
        "check my storage",
        "check storage space",
        "show me disk space",
        "show me storage space",
        "what is my disk usage",
        "whats my disk usage",
        "my disk space",
        "my storage space",
        "my drive space",
        "is my storage almost full",
        "is my storage nearly full",
        "is my disk full",
        "is my drive full",
        "my storage is almost full",
        "my storage is nearly full",
        "my disk is full",
        "my drive is full",
        "running out of disk space",
        "running out of storage",
        "running out of space on my drive",
        "running out of space on my disk",
        "my laptop running out of space",
        "my computer running out of space",
        "whats using up my storage",
        "what is using up my storage"
    });

    const bool memory = has_phrase(text, {
        "how much ram do i have",
        "how much ram is available",
        "how much memory do i have",
        "how much memory is available",
        "available ram on my laptop",
        "available memory on my laptop",
        "ram usage on my laptop",
        "my ram usage",
        "my memory usage",
        "check ram",
        "check my ram",
        "check memory usage",
        "check my memory usage",
        "show me ram",
        "show me memory usage",
        "is my ram full",
        "my ram is full",
        "my laptop running out of ram",
        "my computer running out of ram",
        "my laptop running out of memory",
        "my computer running out of memory",
        "is my laptop running out of memory",
        "is my computer running out of memory",
        "what is my ram usage",
        "whats my ram usage",
        "what is my memory usage",
        "whats my memory usage"
    });

    if (both || (disk && memory)) {
        return Intent::DiskAndMemory;
    }

    if (disk) {
        return Intent::Disk;
    }

    if (memory) {
        return Intent::Memory;
    }

    return Intent::None;
}

} // namespace mikuru_system

#endif
