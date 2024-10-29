#pragma once

#include "MessageSink.h"
#include "utils/stats.h"

#include <string>
#include <string_view>
#include <unordered_set>

namespace dorado {

class Pipeline;

class CorrectionPafReaderNode : public MessageSink {
public:
    CorrectionPafReaderNode(std::string_view paf_file, std::unordered_set<std::string> skip_set);
    ~CorrectionPafReaderNode() = default;
    std::string get_name() const override { return "CorrectionPafReaderNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions&) override {};
    void restart() override {}
    // Main driver function.
    void process(Pipeline& pipeline);

private:
    std::string m_paf_file;
    size_t m_reads_to_infer{0};
    std::unordered_set<std::string> m_skip_set;
};

}  // namespace dorado