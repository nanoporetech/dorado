#pragma once

#include "CorrectionAligner.h"
#include "utils/stats.h"

namespace dorado {

class Pipeline;

class CorrectionAligner {
public:
    virtual ~CorrectionAligner() = default;

    virtual std::string get_name() const = 0;
    virtual stats::NamedStats sample_stats() const = 0;

    // Main driver function.
    virtual void process(Pipeline& pipeline) = 0;

protected:
    CorrectionAligner() = default;
    CorrectionAligner(const CorrectionAligner&) = delete;
    CorrectionAligner(CorrectionAligner&&) = delete;
    CorrectionAligner& operator=(const CorrectionAligner&) = delete;
    CorrectionAligner& operator=(CorrectionAligner&&) = delete;
};

}  // namespace dorado