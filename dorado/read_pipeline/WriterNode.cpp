#include "WriterNode.h"

#include "Version.h"

#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

void WriterNode::worker_thread() {
    if (!m_emit_fastq) {
        std::cout << "@HD\tVN:1.5\tSO:unknown\n"
                  << "@PG\tID:basecaller\tPN:dorado\tVN:" << DORADO_VERSION << "\tCL:dorado";
        for (const auto& arg : m_args) {
            std::cout << " " << arg;
        }
        std::cout << "\n";
    }

    while (true) {
        // Wait until we are provided with a read
        std::unique_lock<std::mutex> lock(m_cv_mutex);
        m_cv.wait_for(lock, 100ms, [this] { return !m_reads.empty(); });
        if (m_reads.empty()) {
            if (m_terminate) {  // Kill the worker if we're done
                return;
            } else {
                continue;
            }
        }

        std::shared_ptr<Read> read = m_reads.front();
        m_reads.pop_front();
        lock.unlock();

        if (m_num_samples_processed >
            std::numeric_limits<std::int64_t>::max() - read->raw_data.size(0)) {
            EXIT_FAILURE;
        }
        m_num_samples_processed += read->raw_data.size(0);
        m_num_reads_processed += 1;

        if (m_emit_fastq) {
            std::cout << "@" << read->read_id << "\n"
                      << read->seq << "\n"
                      << "+\n"
                      << read->qstring << "\n";
        } else {
            try {
                for (const auto& sam_line : read->extract_sam_lines()) {
                    std::cout << sam_line << "\n";
                }
            } catch (const std::exception& ex) {
                std::cerr << ex.what() << "\n";
            }
        }
    }
}

WriterNode::WriterNode(std::vector<std::string> args, bool emit_fastq, size_t max_reads)
        : ReadSink(max_reads),
          m_args(std::move(args)),
          m_emit_fastq(emit_fastq),
          m_num_samples_processed(0),
          m_num_reads_processed(0),
          m_initialization_time(std::chrono::system_clock::now()),
          m_worker(new std::thread(&WriterNode::worker_thread, this)) {}

WriterNode::~WriterNode() {
    terminate();
    m_cv.notify_one();
    m_worker->join();
    auto end_time = std::chrono::system_clock::now();

    auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - m_initialization_time)
                    .count();
    std::cerr << "> Reads basecalled: " << m_num_reads_processed << std::endl;
    std::cerr << "> Samples/s: " << std::scientific << m_num_samples_processed / (duration / 1000.0)
              << std::endl;
}
