#include "StereoDuplexEncoderNode.h"

#include <chrono>

using namespace std::chrono_literals;
using namespace torch::indexing;

namespace dorado {
// Let's make a stub which just consumes reads from its input queue and passes it to its output queue.
// Next step

void StereoDuplexEncoderNode::worker_thread(){
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

        m_sink.push_read(read);
    }
}

StereoDuplexEncoderNode::StereoDuplexEncoderNode(ReadSink &sink)
        : ReadSink(1000),
          m_sink(sink)
{
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
