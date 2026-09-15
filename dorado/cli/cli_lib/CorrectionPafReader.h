#pragma once

#include "CorrectionAligner.h"
#include "utils/stats.h"

#include <string>
#include <string_view>
#include <unordered_set>

namespace dorado {

class Pipeline;

class CorrectionPafReader final : public CorrectionAligner {
public:
    CorrectionPafReader(std::string_view paf_file, std::unordered_set<std::string> skip_set);
    ~CorrectionPafReader() = default;
    std::string get_name() const override { return "CorrectionPafReader"; }
    stats::NamedStats sample_stats() const override;

    // Main driver function.
    void process(Pipeline &pipeline) override;

private:
    CorrectionPafReader(const CorrectionPafReader &) = delete;
    CorrectionPafReader &operator=(const CorrectionPafReader &) = delete;
    CorrectionPafReader(CorrectionPafReader &&) = delete;
    CorrectionPafReader &operator=(CorrectionPafReader &&) = delete;

private:
    std::string m_paf_file;
    size_t m_reads_to_infer{0};
    std::unordered_set<std::string> m_skip_set;
};

}  // namespace dorado