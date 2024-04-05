#pragma once

#include "alignment/Minimap2Aligner.h"
#include "alignment/Minimap2Index.h"
#include "alignment/Minimap2IndexSupportTypes.h"
#include "hts_io/FastxRandomReader.h"
#include "read_pipeline/MessageSink.h"
#include "read_pipeline/messages.h"
#include "utils/stats.h"
#include "utils/types.h"

#include <memory>
#include <string>
#include <vector>

namespace dorado {

class ErrorCorrectionMapperNode : public MessageSink {
public:
    ErrorCorrectionMapperNode(const std::string& index_file, int threads);
    ~ErrorCorrectionMapperNode() { stop_input_processing(); }
    std::string get_name() const override { return "ErrorCorrectionMapperNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions&) override { stop_input_processing(); }
    void restart() override {
        start_input_processing(&ErrorCorrectionMapperNode::input_thread_fn, this);
    }
    alignment::HeaderSequenceRecords get_sequence_records_for_header() const;

private:
    std::string m_index_file;
    void input_thread_fn();

    std::unique_ptr<alignment::Minimap2Aligner> m_aligner;
    std::shared_ptr<alignment::Minimap2Index> m_index;

    CorrectionAlignments extract_alignments(const mm_reg1_t* reg,
                                            int hits,
                                            const hts_io::FastxRandomReader* reader);
};

}  // namespace dorado
