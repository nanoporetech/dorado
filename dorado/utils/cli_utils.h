// Add some utilities for CLI.

#include <algorithm>
#include <cmath>
#include <utility>

namespace dorado {

namespace utils {

// Determine the thread allocation for writer and aligner threads
// in dorado aligner.
static std::pair<int, int> aligner_writer_thread_allocation(int aligner_threads,
                                                            int writer_threads,
                                                            int available_threads,
                                                            float aligner_thread_fraction) {
    if (aligner_threads == 0 && writer_threads == 0) {
        aligner_threads = std::max(
                1, static_cast<int>(std::ceil(aligner_thread_fraction * available_threads)));
        writer_threads = std::max(1, available_threads - aligner_threads);
    } else if (aligner_threads == 0) {
        aligner_threads = available_threads - writer_threads;
    } else if (writer_threads == 0) {
        writer_threads = available_threads - aligner_threads;
    }
    return std::make_pair(aligner_threads, writer_threads);
}

}  // namespace utils

}  // namespace dorado
