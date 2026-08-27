#pragma once

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include <cstdio>
#include <utility>

namespace dorado::utils {

namespace details {

template <class T>
void ignore(const T &) {}

#ifdef _WIN32
inline constexpr char NULL_DEVICE[] = "NUL:";
inline constexpr auto close = _close;
inline constexpr auto dup = _dup;
inline constexpr auto dup2 = _dup2;
inline constexpr auto fileno = _fileno;
#else
inline constexpr char NULL_DEVICE[] = "/dev/null";
using ::close;
using ::dup;
using ::dup2;
using ::fileno;
#endif

}  // namespace details

inline int suppress_stderr() {
    fflush(stderr);
    int fd = details::dup(details::fileno(stderr));
    auto old_stream = freopen(details::NULL_DEVICE, "w", stderr);
    details::ignore(old_stream);
    return fd;
}

inline void restore_stderr(int fd) {
    fflush(stderr);
    auto dup_result = details::dup2(fd, details::fileno(stderr));
    details::ignore(dup_result);
    details::close(fd);
}

inline int suppress_stdout() {
    fflush(stdout);
    int fd = details::dup(details::fileno(stdout));
    auto old_stream = freopen(details::NULL_DEVICE, "w", stdout);
    details::ignore(old_stream);
    return fd;
}

inline void restore_stdout(int fd) {
    fflush(stdout);
    auto dup_result = details::dup2(fd, details::fileno(stdout));
    details::ignore(dup_result);
    details::close(fd);
}

/// <summary>
/// RAII helper to provide scoped stderr suppression
/// </summary>
class [[nodiscard]] SuppressStderr final {
    const int m_fd;

private:
    SuppressStderr(const SuppressStderr &) = delete;
    SuppressStderr &operator=(const SuppressStderr &) = delete;
    SuppressStderr(SuppressStderr &&) = delete;
    SuppressStderr &operator=(SuppressStderr &&) = delete;

public:
    SuppressStderr() : m_fd(suppress_stderr()) {}
    ~SuppressStderr() { restore_stderr(m_fd); }

    /// <summary>
    /// Invoke a function with stderr suppressed
    /// </summary>
    template <typename Func>
    static void invoke(Func &&func) {
        SuppressStderr suppression{};
        std::forward<Func>(func)();
    }
};

/// <summary>
/// RAII helper to provide scoped stdout suppression
/// </summary>
class [[nodiscard]] SuppressStdout final {
    const int m_fd;

private:
    SuppressStdout(const SuppressStdout &) = delete;
    SuppressStdout &operator=(const SuppressStdout &) = delete;
    SuppressStdout(SuppressStdout &&) = delete;
    SuppressStdout &operator=(SuppressStdout &&) = delete;

public:
    SuppressStdout() : m_fd(suppress_stdout()) {}
    ~SuppressStdout() { restore_stdout(m_fd); }

    /// <summary>
    /// Invoke a function with stdout suppressed
    /// </summary>
    template <typename Func>
    static void invoke(Func &&func) {
        SuppressStdout suppression{};
        std::forward<Func>(func)();
    }
};

}  // namespace dorado::utils
