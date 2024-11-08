#include "CorrectionPafWriterNode.h"

#include "utils/paf_utils.h"

#include <iostream>

namespace dorado {

CorrectionPafWriterNode::CorrectionPafWriterNode() : MessageSink(10000, 1) {}

CorrectionPafWriterNode::~CorrectionPafWriterNode() { stop_input_processing(); }

void CorrectionPafWriterNode::input_thread_fn() {
    Message message;
    while (get_input_message(message)) {
        if (!std::holds_alternative<CorrectionAlignments>(message)) {
            continue;
        }

        const CorrectionAlignments alignments = std::get<CorrectionAlignments>(std::move(message));

        for (size_t i = 0; i < std::size(alignments.qnames); ++i) {
            utils::serialize_to_paf(std::cout, alignments.qnames[i], alignments.read_name,
                                    alignments.overlaps[i], 0, 0, 60, alignments.cigars[i]);
            std::cout << '\n';
        }
    }
}

stats::NamedStats CorrectionPafWriterNode::sample_stats() const {
    return stats::from_obj(m_work_queue);
}

void CorrectionPafWriterNode::terminate(const FlushOptions &) { stop_input_processing(); }

}  // namespace dorado
