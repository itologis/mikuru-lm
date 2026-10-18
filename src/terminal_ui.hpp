#ifndef MIKURU_TERMINAL_UI_HPP
#define MIKURU_TERMINAL_UI_HPP

#include <cstdlib>
#include <iostream>
#include <string>

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace mikuru_ui {

class TerminalUi {
public:
    TerminalUi() : color_enabled_(supports_color()) {}

    void show_banner() const {
        std::cout << paint("Mikuru", "\033[1;96m")
                  << " | local GRU chat\n"
                  << "Known answers first; experimental generation when needed. "
                  << "Chats stay in memory only.\n"
                  << "Commands: /help, /status, /reset, /quit\n";
    }

    void show_prompt() const {
        std::cout << "\n" << paint("You> ", "\033[1;96m") << std::flush;
    }

    void finish_input_line() const {
        std::cout << '\n';
    }

    void show_reply(const std::string& reply) const {
        std::cout << paint("Mikuru> ", "\033[1;92m")
                  << reply << '\n';
    }

    void show_status(const std::string& message) const {
        std::cout << paint("Mikuru [status]> ", "\033[2;96m")
                  << message << '\n';
    }

private:
    bool color_enabled_;

    static bool supports_color() {
        if (std::getenv("NO_COLOR") != nullptr) return false;
#if !defined(_WIN32)
        return ::isatty(STDOUT_FILENO) != 0;
#else
        return false;
#endif
    }

    std::string paint(
        const std::string& text,
        const char* color
    ) const {
        if (!color_enabled_) return text;
        return std::string(color) + text + "\033[0m";
    }
};

} // namespace mikuru_ui

#endif
