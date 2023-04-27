// Add some utilities for CLI.

#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <utility>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace dorado {

namespace utils {

// Determine the thread allocation for writer and aligner threads
// in dorado aligner.
static std::pair<int, int> aligner_writer_thread_allocation(int available_threads,
                                                            float writer_thread_fraction) {
    // clamping because we need at least 1 thread for alignment and for writing.
    int writer_threads =
            std::clamp(static_cast<int>(std::floor(writer_thread_fraction * available_threads)), 1,
                       available_threads - 1);
    int aligner_threads = std::clamp(available_threads - writer_threads, 1, available_threads - 1);
    return std::make_pair(aligner_threads, writer_threads);
}

static bool is_fd_tty(FILE* fd) {
#ifdef _WIN32
    return _isatty(_fileno(fd));
#else
    return isatty(fileno(fd));
#endif
}

}  // namespace utils

}  // namespace dorado
