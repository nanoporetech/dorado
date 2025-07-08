#include "read_pipeline/nodes/WriterNode.h"

#include "hts_writer/HtsFileWriter.h"
#include "hts_writer/interface.h"
#include "read_pipeline/base/messages.h"
#include "utils/stats.h"

#include <utility>
#include <variant>

namespace dorado {

WriterNode::WriterNode(std::vector<std::unique_ptr<hts_writer::IWriter>> writers)
        : MessageSink(10000, 1), m_writers(std::move(writers)), m_num_writers(m_writers.size()) {
    for (const auto &writer : m_writers) {
        writer->init();
    }
}

WriterNode::~WriterNode() { stop_input_processing(); }

void WriterNode::input_thread_fn() {
    Message message;
    while (get_input_message(message)) {
        if (std::holds_alternative<BamMessage>(message)) {
            const auto &bam_message = std::get<BamMessage>(message);
            auto item = std::ref(bam_message.data);
            for (const auto &writer : m_writers) {
                writer->process(item);
            }
        }
        // As this is the terminal writer node we must not send message onwards
        // send_message_to_sink(std::move(message));
    }
}

stats::NamedStats WriterNode::sample_stats() const {
    stats::NamedStats stats = MessageSink::sample_stats();
    stats["num_writers"] = m_num_writers;

    for (const auto &writer : m_writers) {
        stats.merge(stats::from_obj(*writer.get()));
    }

    return stats;
}

void WriterNode::terminate(const FlushOptions &) {
    // Finish processing all messages before shutdown
    stop_input_processing();
    for (const auto &writer : m_writers) {
        writer->shutdown();
    }
}

void WriterNode::set_hts_file_header(SamHdrPtr hdr) const {
    SamHdrSharedPtr shared_header(std::move(hdr));
    for (const auto &writer : m_writers) {
        if (auto w = dynamic_cast<hts_writer::HtsFileWriter *>(writer.get())) {
            w->set_header(shared_header);
        }
    }
}

}  // namespace dorado
