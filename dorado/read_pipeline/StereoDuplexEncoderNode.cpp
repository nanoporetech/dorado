#include "StereoDuplexEncoderNode.h"

#include <chrono>

using namespace std::chrono_literals;
using namespace torch::indexing;

namespace dorado {
// Let's make a stub which just consumes reads from its input queue and passes it to its output queue.
// Next step

void StereoDuplexEncoderNode::worker_thread() {
    while (true) {
        // Wait until we are provided with a read
        std::unique_lock<std::mutex> lock(m_cv_mutex);
        m_cv.wait_for(lock, 100ms, [this] { return !m_reads.empty(); });
        if (m_reads.empty()) {
            if (m_terminate) {
                // Termination flag is set and read input queue is empty, so terminate the worker
                return;
            } else {
                continue;
            }
        }

        std::shared_ptr<Read> read = m_reads.front();
        m_reads.pop_front();
        lock.unlock();

        bool read_is_template = false;
        bool read_is_complement = false;

        // Check if read is a template with corresponding complement, add it to the store if it is.
        if (m_template_complement_map.find(read->read_id) == m_template_complement_map.end()) {
            read_is_template = true;
            m_reads[read->read_id] = read;
        }

        // Check if read is a complement with a corresponding template, add it to the store if it is.
        if (m_complement_template_map.find(read->read_id) == m_complement_template_map.end()) {
        }

        // Now that the read is in the store, see if its partner is present, if it is, encode the pair into a new read.

        m_sink.push_read(read);
    }
}

StereoDuplexEncoderNode::StereoDuplexEncoderNode(
        ReadSink &sink,
        std::map<std::string, std::string> template_complement_map)
        : ReadSink(1000), m_sink(sink), m_template_complement_map(template_complement_map) {
    // Set up teh complement_template_map

    for (auto key : template_complement_map) {
        m_complement_template_map[key.second] = key.first;
    }

    std::unique_ptr<std::thread> stereo_encoder_worker_thread =
            std::make_unique<std::thread>(&StereoDuplexEncoderNode::worker_thread, this);
    worker_threads.push_back(std::move(stereo_encoder_worker_thread));
}

StereoDuplexEncoderNode::~StereoDuplexEncoderNode() {
    terminate();
    m_cv.notify_one();
    for (auto &t : worker_threads) {
        t->join();
    }
}

}  // namespace dorado
