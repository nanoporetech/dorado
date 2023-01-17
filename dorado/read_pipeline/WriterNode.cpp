#include "WriterNode.h"

#include "Version.h"
#include "utils/sequence_utils.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>

#ifndef _WIN32
#include <unistd.h>
#endif

using namespace std::chrono_literals;

namespace dorado {

void WriterNode::print_header() {
    if (!m_emit_fastq) {
        std::cout << "@HD\tVN:1.6\tSO:unknown\n"
                  << "@PG\tID:basecaller\tPN:dorado\tVN:" << DORADO_VERSION << "\tCL:dorado";
        for (const auto& arg : m_args) {
            std::cout << " " << arg;
        }
        std::cout << "\n";
    }
}

void WriterNode::worker_thread() {
    while (true) {
        // Wait until we are provided with a read
        std::unique_lock<std::mutex> read_lock(m_cv_mutex);
        m_cv.wait_for(read_lock, 100ms, [this] { return !m_reads.empty(); });
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
        read_lock.unlock();

        if (m_num_samples_processed >
            std::numeric_limits<std::int64_t>::max() - read->raw_data.size(0)) {
            EXIT_FAILURE;
        }

        m_num_bases_processed += read->seq.length();
        m_num_samples_processed += read->raw_data.size(0);
        m_num_reads_processed += 1;

        if (m_rna) {
            std::reverse(read->seq.begin(), read->seq.end());
            std::reverse(read->qstring.begin(), read->qstring.end());
        }

        if (m_num_reads_processed % 100 == 0 && m_isatty) {
            std::scoped_lock<std::mutex> lock(m_cerr_mutex);
            std::cerr << "\r> Reads processed: " << m_num_reads_processed;
        }

        if (utils::mean_qscore_from_qstring(read->qstring) < m_min_qscore) {
            m_num_reads_failed += 1;
            continue;
        }

        if (m_emit_fastq) {
            std::scoped_lock<std::mutex> lock(m_cout_mutex);
            std::cout << "@" << read->read_id << "\n"
                      << read->seq << "\n"
                      << "+\n"
                      << read->qstring << "\n";
        } else {
            try {
                for (const auto& sam_line : read->extract_sam_lines(m_emit_moves, m_duplex)) {
                    std::scoped_lock<std::mutex> lock(m_cout_mutex);
                    std::cout << sam_line << "\n";
                }
            } catch (const std::exception& ex) {
                std::scoped_lock<std::mutex> lock(m_cerr_mutex);
                spdlog::error("{}", ex.what());
            }
        }
    }
}

WriterNode::WriterNode(std::vector<std::string> args,
                       bool emit_fastq,
                       bool emit_moves,
                       bool rna,
                       bool duplex,
                       size_t min_qscore,
                       size_t num_worker_threads,
                       size_t max_reads)
        : ReadSink(max_reads),
          m_args(std::move(args)),
          m_emit_fastq(emit_fastq),
          m_emit_moves(emit_moves),
          m_rna(rna),
          m_duplex(duplex),
          m_min_qscore(min_qscore),
          m_num_bases_processed(0),
          m_num_samples_processed(0),
          m_num_reads_processed(0),
          m_num_reads_failed(0),
          m_initialization_time(std::chrono::system_clock::now()) {
#ifdef _WIN32
    m_isatty = true;
#else
    m_isatty = isatty(fileno(stderr));
#endif

    print_header();
    for (size_t i = 0; i < num_worker_threads; i++) {
        m_workers.push_back(
                std::make_unique<std::thread>(std::thread(&WriterNode::worker_thread, this)));
    }
}

WriterNode::~WriterNode() {
    terminate();
    m_cv.notify_one();
    for (auto& m : m_workers) {
        m->join();
    }
    auto end_time = std::chrono::system_clock::now();

    auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - m_initialization_time)
                    .count();

    if (m_isatty) {
        std::cerr << "\r";
    }
    spdlog::info("> Reads basecalled: {}", m_num_reads_processed);
    if (m_min_qscore > 0) {
        spdlog::info("> Reads skipped (qscore < {}): {}", m_min_qscore, m_num_reads_failed);
    }
    std::ostringstream samples_sec;
    if (m_duplex) {
        samples_sec << std::scientific << m_num_bases_processed / (duration / 1000.0);
        spdlog::info("> Bases/s: {}", samples_sec.str());
    } else {
        samples_sec << std::scientific << m_num_samples_processed / (duration / 1000.0);
        spdlog::info("> Samples/s: {}", samples_sec.str());
    }
}

}  // namespace dorado
