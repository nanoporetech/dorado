#include "read_pipeline/nodes/CorrectionPafWriterNode.h"

#include "utils/paf_utils.h"

#include <iostream>

namespace dorado {

CorrectionPafWriterNode::CorrectionPafWriterNode() : MessageSink(10000, 1) {}

CorrectionPafWriterNode::~CorrectionPafWriterNode() {
    stop_input_processing(utils::AsyncQueueTerminateFast::Yes);
}

std::string CorrectionPafWriterNode::get_name() const { return "CorrectionPafWriterNode"; }

void CorrectionPafWriterNode::input_thread_fn() {
    Message message;
    while (get_input_message(message)) {
        const auto *alignments_ptr = std::get_if<CorrectionAlignmentsPtr>(&message);
        if (!alignments_ptr) {
            continue;
        }

        const CorrectionAlignments &alignments = **alignments_ptr;
        for (size_t i = 0; i < std::size(alignments.qnames); ++i) {
            utils::serialize_to_paf(std::cout, alignments.qnames[i], alignments.read_name,
                                    alignments.overlaps[i], 0, 0, 60, alignments.cigars[i]);
            std::cout << '\n';
        }
    }
}

void CorrectionPafWriterNode::terminate(const TerminateOptions &terminate_options) {
    stop_input_processing(terminate_options.fast);
}

void CorrectionPafWriterNode::restart() {
    start_input_processing([this] { input_thread_fn(); }, "paf_writer");
}

}  // namespace dorado
