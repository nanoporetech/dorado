#include "ErrorCorrectionPafWriterNode.h"

#include "read_pipeline/ReadPipeline.h"
#include "utils/alignment_utils.h"
#include "utils/paf_utils.h"
#include "utils/sequence_utils.h"

#include <minimap.h>

#include <cassert>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace dorado {

ErrorCorrectionPafWriterNode::ErrorCorrectionPafWriterNode() : MessageSink(10000, 1) {}

ErrorCorrectionPafWriterNode::~ErrorCorrectionPafWriterNode() { stop_input_processing(); }

void ErrorCorrectionPafWriterNode::input_thread_fn() {
    Message message;
    while (get_input_message(message)) {
        if (!std::holds_alternative<CorrectionAlignments>(message)) {
            continue;
        }

        const auto alignments = std::get<CorrectionAlignments>(std::move(message));

        for (size_t i = 0; i < alignments.qnames.size(); i++) {
            utils::serialize_to_paf(std::cout, alignments.qnames[i], alignments.read_name,
                                    alignments.overlaps[i], 0, 0, 60, alignments.cigars[i]);
            std::cout << std::endl;
        }
    }
}

stats::NamedStats ErrorCorrectionPafWriterNode::sample_stats() const {
    auto stats = stats::from_obj(m_work_queue);
    return stats;
}

void ErrorCorrectionPafWriterNode::terminate(const FlushOptions &) { stop_input_processing(); }

}  // namespace dorado
