#pragma once

#include "SimpleProgressBar.h"
#include "utils/stats.h"

#include <string>

namespace dorado::variant {

class VariantProgressTracker {
public:
    VariantProgressTracker();

    ~VariantProgressTracker();

    void set_description(const std::string& desc);

    void update_progress_bar(const stats::NamedStats& stats);

    void finalize();

private:
    void internal_set_progress(double progress);

    SimpleProgressBar m_progress_bar;

    double m_last_progress_written = -1.;
};

}  // namespace dorado::variant
