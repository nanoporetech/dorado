#include "read_pipeline/nodes/WriterNode.h"

#include "hts_writer/HtsFileWriter.h"
#include "hts_writer/interface.h"
#include "read_pipeline/base/messages.h"
#include "utils/stats.h"

#include <utility>
#include <variant>

namespace dorado {

WriterNode::WriterNode(std::vector<std::unique_ptr<hts_writer::IWriter>> writers)
        : MessageSink(10000, 1), m_writers(std::move(writers)) {
    for (const auto &writer : m_writers) {
        writer->init();
    }
}

WriterNode::~WriterNode() { stop_input_processing(); }

void WriterNode::input_thread_fn() {
    Message message;
    while (get_input_message(message)) {
        m_num_received++;
        if (std::holds_alternative<BamMessage>(message)) {
            m_num_dispatched++;

            const auto &bam_message = std::get<BamMessage>(message);
            auto item = std::ref(bam_message.data);
            for (const auto &writer : m_writers) {
                writer->process(item);
            }
        }
        // send_message_to_sink(std::move(message));
    }
}

stats::NamedStats WriterNode::sample_stats() const {
    stats::NamedStats stats = MessageSink::sample_stats();
    stats["num_writers"] = m_writers.size();
    stats["num_received"] = m_num_received.load();
    stats["num_dispatched"] = m_num_dispatched.load();

    for (const auto &writer : m_writers) {
        for (const auto &[key, value] : stats::from_obj(*writer.get())) {
            stats[key] = value;
        }
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

void WriterNode::take_hts_header(SamHdrPtr hdr) const {
    SamHdrSharedPtr shared_header(std::move(hdr));
    for (const auto &writer : m_writers) {
        if (auto w = dynamic_cast<hts_writer::HtsFileWriter *>(writer.get())) {
            w->set_header(shared_header);
        }
    }
}

}  // namespace dorado
