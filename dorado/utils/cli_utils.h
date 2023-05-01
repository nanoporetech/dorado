// Add some utilities for CLI.

#include "Version.h"

#include <htslib/sam.h>
#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
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

static void add_pg_hdr(sam_hdr_t* hdr, const std::vector<std::string>& args) {
    sam_hdr_add_lines(hdr, "@HD\tVN:1.6\tSO:unknown", 0);

    std::stringstream pg;
    pg << "@PG\tID:basecaller\tPN:dorado\tVN:" << DORADO_VERSION << "\tCL:dorado";
    for (const auto& arg : args) {
        pg << " " << arg;
    }
    pg << std::endl;
    sam_hdr_add_lines(hdr, pg.str().c_str(), 0);
}

}  // namespace utils

}  // namespace dorado
