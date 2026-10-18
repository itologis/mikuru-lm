#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <system_error>

namespace {

std::string format_bytes(unsigned long long bytes) {
    constexpr const char* units[] = {
        "B", "KiB", "MiB", "GiB", "TiB"
    };

    double value = static_cast<double>(bytes);
    int unit = 0;

    while (value >= 1024.0 && unit < 4) {
        value /= 1024.0;
        ++unit;
    }

    std::ostringstream output;

    if (unit == 0) {
        output << bytes << ' ' << units[unit];
    } else {
        output << std::fixed << std::setprecision(1)
               << value << ' ' << units[unit];
    }

    return output.str();
}

void show_disk_space() {
    std::error_code error;
    const auto info = std::filesystem::space("/", error);

    if (error) {
        std::cout << "Disk inspection failed: "
                  << error.message() << '\n';
        return;
    }

    const auto used = info.capacity >= info.free
        ? info.capacity - info.free
        : 0ULL;

    const double percentage = info.capacity == 0
        ? 0.0
        : 100.0 * static_cast<double>(used)
              / static_cast<double>(info.capacity);

    std::cout << "Root filesystem disk space\n"
              << "  Available: " << format_bytes(info.available) << '\n'
              << "  Total:     " << format_bytes(info.capacity) << '\n'
              << "  Occupied:  " << std::fixed
              << std::setprecision(1) << percentage << "%\n";
}

void show_memory_usage() {
    std::ifstream file("/proc/meminfo");

    if (!file) {
        std::cout << "RAM inspection failed: cannot read /proc/meminfo.\n";
        return;
    }

    unsigned long long total_kib = 0;
    unsigned long long available_kib = 0;

    std::string line;

    while (std::getline(file, line)) {
        std::istringstream fields(line);
        std::string key;
        unsigned long long value = 0;

        fields >> key >> value;

        if (key == "MemTotal:") {
            total_kib = value;
        } else if (key == "MemAvailable:") {
            available_kib = value;
        }
    }

    if (total_kib == 0 || available_kib > total_kib) {
        std::cout << "RAM inspection failed: incomplete memory data.\n";
        return;
    }

    const auto total_bytes = total_kib * 1024ULL;
    const auto available_bytes = available_kib * 1024ULL;
    const auto used_bytes = total_bytes - available_bytes;

    const double percentage =
        100.0 * static_cast<double>(used_bytes)
        / static_cast<double>(total_bytes);

    std::cout << "System memory (RAM)\n"
              << "  Available: " << format_bytes(available_bytes) << '\n'
              << "  Total:     " << format_bytes(total_bytes) << '\n'
              << "  In use:    " << std::fixed
              << std::setprecision(1) << percentage << "%\n";
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout
            << "Mikuru System Inspector v0.1\n"
            << "Usage: ./build/mikuru-system-inspector disk|memory|all\n";
        return 0;
    }

    const std::string command = argv[1];

    if (command == "disk") {
        show_disk_space();
    } else if (command == "memory") {
        show_memory_usage();
    } else if (command == "all") {
        show_disk_space();
        std::cout << '\n';
        show_memory_usage();
    } else {
        std::cerr
            << "Unknown command: " << command << '\n'
            << "Choose disk, memory, or all.\n";
        return 1;
    }

    return 0;
}
