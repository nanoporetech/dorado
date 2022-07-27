#include "ModBaseCallerNode.h"

#include "nn/RemoraModel.h"
#include "utils/math_utils.h"

#include <chrono>
using namespace std::chrono_literals;

ModBaseCallerNode::ModBaseCallerNode(ReadSink& sink,
                                     std::shared_ptr<RemoraRunner> model_runner,
                                     size_t model_stride,
                                     size_t max_reads)
        : ReadSink(max_reads),
          m_sink(sink),
          m_model_runner(model_runner),
          m_model_stride(model_stride),
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

        // TODO: better read queuing, multiple runners
        auto base_mod_probs =
                m_model_runner->run(read->raw_data, read->seq, read->moves, m_model_stride);
        read->base_mod_probs.resize(base_mod_probs.size(0) * base_mod_probs.size(1));
        auto results_view =
                base_mod_probs.view({static_cast<int64_t>(read->base_mod_probs.size())});
        // Rescale network outputs (0-1) to SAM format (0-255)
        for (auto i = 0ul; i < read->base_mod_probs.size(); ++i) {
            read->base_mod_probs[i] =
                    uint8_t(std::min(std::floor(results_view[i].item().toFloat() * 256), 255.0f));
        }

        read->base_mod_info = m_model_runner->base_mod_info();

        // Pass the read to the next node
        m_sink.push_read(read);
    }
}
