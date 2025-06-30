#include "read_pipeline/nodes/WriterNode.h"

#include "hts_writer/interface.h"
#include "read_pipeline/base/messages.h"

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
        send_message_to_sink(std::move(message));
    }
}

stats::NamedStats WriterNode::sample_stats() const {
    stats::NamedStats stats = MessageSink::sample_stats();
    stats["num_writers"] = m_writers.size();
    stats["num_received"] = m_num_received.load();
    stats["num_dispatched"] = m_num_dispatched.load();
    return stats;
}

void WriterNode::terminate(const FlushOptions &) {
    for (const auto &writer : m_writers) {
        writer->shutdown();
    }
    stop_input_processing();
}

}  // namespace dorado
