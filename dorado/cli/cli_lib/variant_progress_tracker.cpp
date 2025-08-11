#include "variant_progress_tracker.h"

#include "utils/tty_utils.h"

#include <spdlog/spdlog.h>

#include <algorithm>

namespace dorado::variant {

VariantProgressTracker::VariantProgressTracker() = default;

VariantProgressTracker::~VariantProgressTracker() = default;

void VariantProgressTracker::set_description(const std::string& desc) {
    m_progress_bar.erase_progress_bar_line();
    m_progress_bar.set_option(indicators::option::PostfixText{desc});
}

void VariantProgressTracker::update_progress_bar(const stats::NamedStats& stats) {
    const auto fetch_stat = [&stats](const std::string& name) {
        auto res_stats = stats.find(name);
        if (res_stats != std::cend(stats)) {
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

void VariantProgressTracker::finalize() {
    internal_set_progress(100.0);
    std::cerr << '\n';  // Keep the progress bar.
}

void VariantProgressTracker::internal_set_progress(double progress) {
    // The progress bar uses escape sequences that only TTYs understand.
    if (!utils::is_fd_tty(stderr)) {
        return;
    }

    // Sanity clamp.
    progress = std::min(progress, 100.);

    // Draw it.
    std::cerr << '\r';
    m_progress_bar.set_progress(progress);
}

}  // namespace dorado::variant
