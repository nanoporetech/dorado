// Add some utilities for CLI.
#pragma once

#include <cctype>
#include <iostream>
#include <utility>
#ifdef _WIN32
#include <io.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace dorado {

namespace utils {

inline bool is_fd_tty(FILE* fd) {
#ifdef _WIN32
    return _isatty(_fileno(fd));
#else
    return isatty(fileno(fd));
#endif
}

#ifdef _WIN32
inline bool is_fd_pipe(FILE*) { return false; }
#else
inline bool is_fd_pipe(FILE* fd) {
    struct stat buffer;
    fstat(fileno(fd), &buffer);
    return S_ISFIFO(buffer.st_mode);
}
#endif

}  // namespace utils

}  // namespace dorado
