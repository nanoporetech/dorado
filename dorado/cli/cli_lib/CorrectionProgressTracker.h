#pragma once

#include "SimpleProgressBar.h"
#include "utils/stats.h"

#include <cstdint>
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

    SimpleProgressBar m_progress_bar;

    float m_last_progress_written = -1.f;
};

}  // namespace dorado
