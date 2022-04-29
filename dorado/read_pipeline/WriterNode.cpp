#include <chrono>
#include <iostream>
#include "WriterNode.h"

using namespace std::chrono_literals;

void WriterNode::worker_thread() {

    while (true) {
        // Wait until we are provided with a read
        std::unique_lock<std::mutex> lock(m_cv_mutex);
        m_cv.wait_for(lock, 100ms, [this] {return !m_reads.empty(); });
        if (m_reads.empty()) {
            if (m_terminate) { // Kill the worker if we're done
                return;
            }
            else {
                continue;
            }
        }

        std::shared_ptr<Read> read = m_reads.front();
        m_reads.pop_front();
        lock.unlock();

        num_samples_processed += read->raw_data.size(0);
        num_reads_processed += 1;

        std::cout << "@" << read->read_id << "\n"
                  << read->seq << "\n"
                  << "+\n"
                  << read->qstring << "\n";

    }
}

WriterNode::WriterNode(size_t max_reads) {
    m_max_reads = max_reads;
    num_samples_processed = 0;
    num_reads_processed = 0;
    initialization_time = std::chrono::system_clock::now();
    m_worker.reset(new std::thread(&WriterNode::worker_thread, this));
}

WriterNode::~WriterNode() {
    m_worker->join();
    auto end_time = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - initialization_time).count();
    std::cerr << "> Reads basecalled: " << num_reads_processed << std::endl;
    std::cerr << "> Samples/s: " << std::scientific << num_samples_processed / (duration / 1000.0) << std::endl;
}
