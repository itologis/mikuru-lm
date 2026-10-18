#ifndef MIKURU_SYSTEM_PERSONALITY_HPP
#define MIKURU_SYSTEM_PERSONALITY_HPP

#include "intent_router.hpp"

#include <iomanip>
#include <sstream>
#include <string>

namespace mikuru_system {

inline bool personality_field(
    const std::string& report,
    const std::string& section_title,
    const std::string& field_name,
    std::string& value) {

    const auto start = report.find(section_title);
    if (start == std::string::npos) {
        return false;
    }

    const auto section_end = report.find("\n\n", start);
    const std::string section = report.substr(
        start,
        section_end == std::string::npos
            ? std::string::npos
            : section_end - start);

    const auto field = section.find(field_name);
    if (field == std::string::npos) {
        return false;
    }

    auto value_start = field + field_name.size();

    while (value_start < section.size() &&
           (section[value_start] == ' ' ||
            section[value_start] == '\t')) {
        ++value_start;
    }

    auto value_end = section.find('\n', value_start);
    if (value_end == std::string::npos) {
        value_end = section.size();
    }

    value = section.substr(value_start, value_end - value_start);

    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return false;
    }

    const auto last = value.find_last_not_of(" \t\r\n");
    value = value.substr(first, last - first + 1);

    return !value.empty();
}

inline bool personality_percentage(
    const std::string& report,
    const std::string& section_title,
    const std::string& field_name,
    double& percentage) {

    std::string value;

    if (!personality_field(
            report, section_title, field_name, value)) {
        return false;
    }

    try {
        percentage = std::stod(value);
        return percentage >= 0.0 && percentage <= 100.0;
    } catch (...) {
        return false;
    }
}

inline std::string personality_disk_message(
    const std::string& report) {

    std::string available;
    std::string total;
    double occupied = 0.0;

    if (!personality_field(
            report, "Root filesystem disk space",
            "Available:", available) ||
        !personality_field(
            report, "Root filesystem disk space",
            "Total:", total) ||
        !personality_percentage(
            report, "Root filesystem disk space",
            "Occupied:", occupied)) {
        return "Hmm, I couldn't read the disk details clearly, "
               "so I won't guess. We can check the inspector "
               "before deciding what to do.";
    }

    std::ostringstream output;
    output << "Quick check: your system storage ";

    if (occupied >= 90.0) {
        output << "is getting pretty full. ";
    } else if (occupied >= 80.0) {
        output << "is getting a little tight. ";
    } else {
        output << "has some breathing room. ";
    }

    output << "You've got " << available
           << " available out of " << total
           << ", with " << std::fixed << std::setprecision(1)
           << occupied << "% occupied. ";

    if (occupied >= 90.0) {
        output << "I'd hold off on big downloads for now. "
               << "We can check what's taking up the space next, "
               << "without deleting anything.";
    } else if (occupied >= 80.0) {
        output << "I'd keep an eye on large downloads. "
               << "We can look for space-saving options together "
               << "before removing anything.";
    } else {
        output << "Nothing here suggests an urgent storage problem. "
               << "We can check the larger folders whenever you "
               << "need to free up more space.";
    }

    return output.str();
}

inline std::string personality_memory_message(
    const std::string& report) {

    std::string available;
    std::string total;
    double in_use = 0.0;

    if (!personality_field(
            report, "System memory (RAM)",
            "Available:", available) ||
        !personality_field(
            report, "System memory (RAM)",
            "Total:", total) ||
        !personality_percentage(
            report, "System memory (RAM)",
            "In use:", in_use)) {
        return "Hmm, I couldn't read the RAM details clearly, "
               "so I won't guess. We can check the inspector "
               "before drawing conclusions.";
    }

    std::ostringstream output;
    output << "Quick check: your RAM ";

    if (in_use >= 85.0) {
        output << "is pretty busy right now. ";
    } else if (in_use >= 65.0) {
        output << "is moderately occupied right now. ";
    } else {
        output << "has some breathing room. ";
    }

    output << "About " << std::fixed << std::setprecision(1)
           << in_use << "% is in use, with " << available
           << " available out of " << total << ". ";

    if (in_use >= 85.0) {
        output << "That's worth checking if your apps feel sluggish, "
               << "though this number alone doesn't prove there's "
               << "a problem. We can inspect which processes use "
               << "the most memory next.";
    } else if (in_use >= 65.0) {
        output << "Linux uses memory for caching too, so this "
               << "percentage alone doesn't mean anything is wrong. "
               << "If the laptop feels slow, we can investigate "
               << "what's using memory.";
    } else {
        output << "Looks comfortable from this reading. No need "
               << "to change anything just because some RAM is in use.";
    }

    return output.str();
}

inline std::string format_personality_response(
    Intent intent,
    const std::string& report) {

    switch (intent) {
        case Intent::Disk:
            return personality_disk_message(report);

        case Intent::Memory:
            return personality_memory_message(report);

        case Intent::DiskAndMemory:
            return personality_disk_message(report) +
                   "\n\n" + personality_memory_message(report);

        case Intent::None:
            break;
    }

    return "I couldn't identify which system information you "
           "wanted, so I haven't run an inspection.";
}

} // namespace mikuru_system

#endif
