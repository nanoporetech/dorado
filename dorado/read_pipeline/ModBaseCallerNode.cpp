#include "ModBaseCallerNode.h"

#include <chrono>
using namespace std::chrono_literals;

ModBaseCallerNode::ModBaseCallerNode(ReadSink& sink,
                                     std::vector<std::shared_ptr<RemoraRunner>>& model_runners,
                                     size_t max_reads)
        : ReadSink(max_reads),
          m_sink(sink),
          m_worker(new std::thread(&ModBaseCallerNode::worker_thread, this)) {}

ModBaseCallerNode::~ModBaseCallerNode() {
    terminate();
    m_cv.notify_one();
    m_worker->join();
}

void ModBaseCallerNode::worker_thread() {
    while (true) {
        // Wait until we are provided with a read
        std::unique_lock<std::mutex> lock(m_cv_mutex);
        m_cv.wait_for(lock, 100ms, [this] { return !m_reads.empty(); });
        if (m_reads.empty()) {
            if (m_terminate) {
                // Notify our sink and then kill the worker if we're done
                m_sink.terminate();
                return;
            } else {
                continue;
            }
        }

        std::shared_ptr<Read> read = m_reads.front();
        m_reads.pop_front();
        lock.unlock();

        // Pass the read to the next node
        m_sink.push_read(read);
    }
}
