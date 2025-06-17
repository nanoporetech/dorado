#pragma once

#include "utils/stats.h"

#ifdef _WIN32
#include <indicators/progress_bar.hpp>
#else
#include <indicators/block_progress_bar.hpp>
#endif

#include <string>

namespace dorado::polisher {

class PolishProgressTracker {
public:
    PolishProgressTracker();

    ~PolishProgressTracker();

    void set_description(const std::string& desc);

    void update_progress_bar(const stats::NamedStats& stats);

    void finalize();

private:
    void internal_set_progress(double progress);

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

    double m_last_progress_written = -1.;
};

}  // namespace dorado::polisher
