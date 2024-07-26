#ifndef DORADO_READ_PIPELINE_ERROR_CORRECTION_PAF_READER_NODE_H
#define DORADO_READ_PIPELINE_ERROR_CORRECTION_PAF_READER_NODE_H

#include "ReadPipeline.h"
#include "read_pipeline/MessageSink.h"
#include "read_pipeline/flush_options.h"
#include "read_pipeline/messages.h"
#include "utils/stats.h"

#include <string>
#include <string_view>

namespace dorado {

class ErrorCorrectionPafReaderNode : public MessageSink {
public:
    ErrorCorrectionPafReaderNode(std::string_view paf_file);
    ~ErrorCorrectionPafReaderNode() = default;
    std::string get_name() const override { return "ErrorCorrectionPafReaderNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions&) override{};
    void restart() override {}
    // Main driver function.
    void process(Pipeline& pipeline);

private:
    std::string m_paf_file;

    size_t m_reads_to_correct;
};

}  // namespace dorado

#endif  // DORADO_READ_PIPELINE_ERROR_CORRECTION_PAF_READER_NODE_H