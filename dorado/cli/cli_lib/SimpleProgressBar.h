#pragma once

#include "utils/tty_utils.h"

#ifdef _WIN32
#include <indicators/progress_bar.hpp>
#else
#include <indicators/block_progress_bar.hpp>
#endif

namespace dorado {

#ifdef _WIN32
using ProgressBarBase = indicators::ProgressBar;
#else
using ProgressBarBase = indicators::BlockProgressBar;
#endif

class SimpleProgressBar : public ProgressBarBase {
public:
    SimpleProgressBar()
            : ProgressBarBase{
                      indicators::option::Stream{std::cerr},
                      indicators::option::BarWidth{30},
                      indicators::option::ShowElapsedTime{true},
                      indicators::option::ShowRemainingTime{true},
                      indicators::option::ShowPercentage{true},
              } {}

    void set_progress(float value) {
        // The progress bar uses escape sequences that only TTYs understand.
        if (dorado::utils::is_fd_tty(stderr)) {
#ifdef _WIN32
            ProgressBarBase::set_progress(static_cast<size_t>(value));
#else
            ProgressBarBase::set_progress(value);
#endif
        }
    }

    void erase_progress_bar_line() const {
        // Don't write escape codes unless it's a TTY.
        if (dorado::utils::is_fd_tty(stderr)) {
            // Erase the current line so that we remove the previous description.
#ifndef _WIN32
            // I would use indicators::erase_progress_bar_line() here, but it hardcodes stdout.
            std::cerr << "\r\033[K";
#endif
        }
    }
};

}  // namespace dorado
