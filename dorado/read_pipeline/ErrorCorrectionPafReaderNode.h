#pragma once

#include "ReadPipeline.h"
#include "alignment/Minimap2Aligner.h"
#include "alignment/Minimap2Index.h"
#include "alignment/Minimap2IndexSupportTypes.h"
#include "read_pipeline/MessageSink.h"
#include "read_pipeline/messages.h"
#include "utils/AsyncQueue.h"
#include "utils/stats.h"
#include "utils/types.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dorado {

class ErrorCorrectionPafReaderNode : public MessageSink {
public:
    ErrorCorrectionPafReaderNode(const std::string& paf_file);
    ~ErrorCorrectionPafReaderNode() = default;
    std::string get_name() const override { return "ErrorCorrectionPafReaderNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions&) override{};
    void restart() override {}
    // Main driver function.
    void process(Pipeline& pipeline);

private:
    std::string m_paf_file;

    std::unordered_map<std::string, CorrectionAlignments> m_correction_records;

    size_t m_reads_to_correct;
};

}  // namespace dorado
