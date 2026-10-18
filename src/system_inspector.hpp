#ifndef MIKURU_SYSTEM_INSPECTOR_ROUTER_HPP
#define MIKURU_SYSTEM_INSPECTOR_ROUTER_HPP

#include <cerrno>
#include <cctype>
#include <initializer_list>
#include <string>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "intent_router.hpp"
#include "system_personality.hpp"

namespace mikuru_system {

inline bool answer_system_question(
    const std::string& input, std::string& answer) {

    const Intent intent = classify(input);
    const char* mode = nullptr;

    switch (intent) {
        case Intent::Disk:
            mode = "disk";
            break;
        case Intent::Memory:
            mode = "memory";
            break;
        case Intent::DiskAndMemory:
            mode = "all";
            break;
        case Intent::None:
            return false;
    }

    int pipe_fd[2];

    if (::pipe(pipe_fd) == -1) {
        answer = "I couldn't open a connection to the system inspector.";
        return true;
    }

    const pid_t child = ::fork();

    if (child == -1) {
        ::close(pipe_fd[0]);
        ::close(pipe_fd[1]);
        answer = "I couldn't start the system inspector.";
        return true;
    }

    if (child == 0) {
        ::close(pipe_fd[0]);

        if (::dup2(pipe_fd[1], STDOUT_FILENO) == -1) {
            _exit(126);
        }

        ::close(pipe_fd[1]);

        // No shell and no user-provided command. Execute only the
        // fixed inspector binary with the selected fixed mode.
        ::execl(
            "./build/mikuru-system-inspector",
            "mikuru-system-inspector",
            mode,
            static_cast<char*>(nullptr));

        _exit(127);
    }

    ::close(pipe_fd[1]);

    char buffer[512];

    while (true) {
        const ssize_t count = ::read(
            pipe_fd[0], buffer, sizeof(buffer));

        if (count > 0) {
            answer.append(buffer, static_cast<std::size_t>(count));
        } else if (count == -1 && errno == EINTR) {
            continue;
        } else {
            break;
        }
    }

    ::close(pipe_fd[0]);

    int status = 0;
    pid_t waited;

    do {
        waited = ::waitpid(child, &status, 0);
    } while (waited == -1 && errno == EINTR);

    if (waited == -1 ||
        !WIFEXITED(status) ||
        WEXITSTATUS(status) != 0 ||
        answer.empty()) {
        answer =
            "I couldn't get a result from the system inspector. "
            "Please check that it compiled successfully.";
        return true;
    }

    answer = format_personality_response(intent, answer);
    return true;
}

} // namespace mikuru_system

#endif
