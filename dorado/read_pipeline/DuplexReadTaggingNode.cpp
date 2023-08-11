#include "DuplexReadTaggingNode.h"

#include <spdlog/spdlog.h>

namespace dorado {

void DuplexReadTaggingNode::worker_thread() {
    torch::InferenceMode inference_mode_guard;

    Message message;
    while (get_input_message(message)) {
        // If this message isn't a read, we'll get a bad_variant_access exception.
        auto read = std::get<std::shared_ptr<Read>>(message);

        if (!read->is_duplex && !read->is_duplex_parent) {
            send_message_to_sink(read);
        } else if (read->is_duplex) {
            std::string template_read_id = read->read_id.substr(0, read->read_id.find(';'));
            std::string complement_read_id =
                    read->read_id.substr(read->read_id.find(';') + 1, read->read_id.length());

            send_message_to_sink(read);

            for (auto& rid : {template_read_id, complement_read_id}) {
                if (m_parents_processed.find(rid) != m_parents_processed.end()) {
                    // Parent read has already been processed. Do nothing.
                    continue;
                }
                auto find_parent = m_duplex_parents.find(rid);
                if (find_parent != m_duplex_parents.end()) {
                    // Parent read has been seen. Process it and send it
                    // downstream.
                    send_message_to_sink(find_parent->second);
                    m_parents_processed.insert(rid);
                    m_duplex_parents.erase(find_parent);
                } else {
                    // Parent read hasn't been seen. So add it to list of
                    // parents to look for.
                    m_parents_wanted.insert(rid);
                }
            }
        } else {
            // If a read has already been seen and processed, send it on
            if (m_parents_processed.find(read->read_id) != m_parents_processed.end()) {
                send_message_to_sink(read);
            }
            auto find_read = m_parents_wanted.find(read->read_id);
            if (find_read != m_parents_wanted.end()) {
                send_message_to_sink(read);
                m_parents_processed.insert(read->read_id);
            } else {
                m_duplex_parents[read->read_id] = std::move(read);
            }
        }
    }

    for (auto& [k, v] : m_duplex_parents) {
        v->is_duplex_parent = false;
        send_message_to_sink(v);
    }
}

DuplexReadTaggingNode::DuplexReadTaggingNode() : MessageSink(1000) { start_threads(); }

void DuplexReadTaggingNode::start_threads() {
    m_worker =
            std::make_unique<std::thread>(std::thread(&DuplexReadTaggingNode::worker_thread, this));
}

void DuplexReadTaggingNode::terminate_impl() {
    terminate_input_queue();
    if (m_worker->joinable()) {
        m_worker->join();
    }
}

void DuplexReadTaggingNode::restart() {
    restart_input_queue();
    start_threads();
}

stats::NamedStats DuplexReadTaggingNode::sample_stats() const {
    stats::NamedStats stats = stats::from_obj(m_work_queue);
    return stats;
}

}  // namespace dorado
