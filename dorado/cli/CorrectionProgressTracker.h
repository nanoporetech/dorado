#pragma once

#include "utils/stats.h"

#ifdef _WIN32
#include <indicators/progress_bar.hpp>
#else
#include <indicators/block_progress_bar.hpp>
#endif

#include <chrono>
#include <cstdint>
#include <map>
#include <string>

namespace dorado {

// Collect and calculate progress related
// statistics for error correction.
class CorrectionProgressTracker {
public:
    CorrectionProgressTracker();
    ~CorrectionProgressTracker();

    void set_description(const std::string& desc);

    void summarize() const;
    void update_progress_bar(const stats::NamedStats& stats,
                             const stats::NamedStats& aligner_stats);

private:
    void internal_set_progress(float progress);

    int64_t m_num_reads_corrected{0};

#ifdef _WIN32
    indicators::ProgressBar m_progress_bar {
#else
    indicators::BlockProgressBar m_progress_bar{
#endif
        indicators::option::Stream{std::cerr}, indicators::option::BarWidth{30},
                indicators::option::ShowElapsedTime{true},
                indicators::option::ShowRemainingTime{true},
                indicators::option::ShowPercentage{true},
    };

    float m_last_progress_written = -1.f;
};

}  // namespace dorado
