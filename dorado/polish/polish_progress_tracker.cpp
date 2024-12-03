#include "polish_progress_tracker.h"

#include "utils/string_utils.h"
#include "utils/tty_utils.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace {

void erase_progress_bar_line() {
    // Don't write escape codes unless it's a TTY.
    if (dorado::utils::is_fd_tty(stderr)) {
        // Erase the current line so that we remove the previous description.
#ifndef _WIN32
        // I would use indicators::erase_progress_bar_line() here, but it hardcodes stdout.
        std::cerr << "\r\033[K";
#endif
    }
}

}  // namespace

namespace dorado::polisher {

PolishProgressTracker::PolishProgressTracker() = default;

PolishProgressTracker::~PolishProgressTracker() = default;

void PolishProgressTracker::set_description(const std::string& desc) {
    erase_progress_bar_line();
    m_progress_bar.set_option(indicators::option::PostfixText{desc});
}

void PolishProgressTracker::summarize() const {
    erase_progress_bar_line();

    // if (m_num_polished_sequences > 0) {
    //     spdlog::info("Polished sequences written: {}", m_num_polished_sequences);
    // }
}

void PolishProgressTracker::update_progress_bar(const stats::NamedStats& stats) {
    auto fetch_stat = [&stats](const std::string& name) {
        auto res_stats = stats.find(name);
        if (res_stats != stats.end()) {
            return res_stats->second;
        }
        return 0.;
    };

    const double total = fetch_stat("total");
    const double processed = fetch_stat("processed");
    const double progress = std::min(100.0, 100.0 * processed / total);

    if (progress > m_last_progress_written) {
        m_last_progress_written = progress;
        internal_set_progress(progress);
    } else {
        internal_set_progress(m_last_progress_written);
    }
}

void PolishProgressTracker::internal_set_progress(float progress) {
    // The progress bar uses escape sequences that only TTYs understand.
    if (!utils::is_fd_tty(stderr)) {
        return;
    }

    // Sanity clamp.
    progress = std::min(progress, 100.f);

    // Draw it.
#ifdef _WIN32
    m_progress_bar.set_progress(static_cast<size_t>(progress));
#else
    m_progress_bar.set_progress(progress);
#endif

    if (progress < 100.0) {
        std::cerr << '\r';
    } else {
        std::cerr << '\n';
    }
}

}  // namespace dorado::polisher
