#include "WriterNode.h"

#include "Version.h"
#include "utils/sequence_utils.h"

#include <htslib/sam.h>
#include <indicators/progress_bar.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>

#ifndef _WIN32
#include <unistd.h>
#endif

using namespace std::chrono_literals;

namespace dorado {

void WriterNode::add_header() {
    if (m_emit_fastq) {
        return;  // No header for fastq.
    }

    m_header = sam_hdr_init();
    sam_hdr_add_lines(m_header, "@HD\tVN:1.6\tSO:unknown", 0);

    std::stringstream pg;
    pg << "@PG\tID:basecaller\tPN:dorado\tVN:" << DORADO_VERSION << "\tCL:dorado";
    for (const auto& arg : m_args) {
        pg << " " << arg;
    }
    pg << std::endl;
    sam_hdr_add_lines(m_header, pg.str().c_str(), 0);

    // Add read groups
    for (auto const& x : m_read_groups) {
        std::stringstream rg;
        rg << "@RG\t";
        rg << "ID:" << x.first << "\t";
        rg << "PU:" << x.second.flowcell_id << "\t";
        rg << "PM:" << x.second.device_id << "\t";
        rg << "DT:" << x.second.exp_start_time << "\t";
        rg << "PL:"
           << "ONT"
           << "\t";
        rg << "DS:"
           << "basecall_model=" << x.second.basecalling_model << " runid=" << x.second.run_id
           << "\t";
        rg << "LB:" << x.second.sample_id << "\t";
        rg << "SM:" << x.second.sample_id;
        rg << std::endl;
        sam_hdr_add_lines(m_header, rg.str().c_str(), 0);
    }

    if (sam_hdr_write(m_file, m_header) < 0) {
        throw std::runtime_error("Unable to write the SAM/BAM header.");
    };
}

void WriterNode::worker_thread() {
    m_active_threads++;

    Message message;
    while (m_work_queue.try_pop(message)) {
        // If this message isn't a read, we'll get a bad_variant_access exception.
        auto read = std::get<std::shared_ptr<Read>>(message);

        m_num_bases_processed += read->seq.length();
        m_num_samples_processed += read->raw_data.size(0);
        ++m_num_reads_processed;

        if (m_rna) {
            std::reverse(read->seq.begin(), read->seq.end());
            std::reverse(read->qstring.begin(), read->qstring.end());
        }

        if (((m_num_reads_processed % m_progress_bar_increment) == 0) && m_isatty &&
            ((m_num_reads_processed / m_progress_bar_increment) < 100)) {
            if (m_num_reads_expected != 0) {
                m_progress_bar.tick();
            } else {
                std::scoped_lock<std::mutex> lock(m_cerr_mutex);
                std::cerr << "\r> Reads processed: " << m_num_reads_processed;
            }
        }

        if (m_emit_fastq) {
            std::scoped_lock<std::mutex> lock(m_cout_mutex);
            std::cout << "@" << read->read_id << "\n"
                      << read->seq << "\n"
                      << "+\n"
                      << read->qstring << "\n";
        } else {
            try {
                auto alns = read->extract_sam_lines(m_emit_moves, m_duplex);
                for (const auto aln : alns) {
                    std::scoped_lock<std::mutex> lock(m_cout_mutex);
                    if (sam_write1(m_file, m_header, aln) < 0) {
                        spdlog::warn("Unable to write alignment for read {}", read->read_id);
                    }
                }
            } catch (const std::exception& ex) {
                std::scoped_lock<std::mutex> lock(m_cerr_mutex);
                spdlog::error("{}", ex.what());
            }
        }
    }

    auto num_active_threads = --m_active_threads;
    if (num_active_threads == 0) {
        auto end_time = std::chrono::system_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                                              m_initialization_time)
                                .count();
        if (m_isatty) {
            std::cerr << "\r";
        }
        spdlog::info("> Reads basecalled: {}", m_num_reads_processed);
        std::ostringstream samples_sec;
        if (m_duplex) {
            samples_sec << std::scientific << m_num_bases_processed / (duration / 1000.0);
            spdlog::info("> Bases/s: {}", samples_sec.str());
        } else {
            samples_sec << std::scientific << m_num_samples_processed / (duration / 1000.0);
            spdlog::info("> Samples/s: {}", samples_sec.str());
        }
    }
}

WriterNode::WriterNode(std::vector<std::string> args,
                       bool emit_fastq,
                       bool emit_moves,
                       bool rna,
                       bool duplex,
                       size_t num_worker_threads,
                       std::unordered_map<std::string, ReadGroup> read_groups,
                       int num_reads,
                       size_t max_reads)
        : MessageSink(max_reads),
          m_args(std::move(args)),
          m_emit_fastq(emit_fastq),
          m_emit_moves(emit_moves),
          m_rna(rna),
          m_duplex(duplex),
          m_read_groups(std::move(read_groups)),
          m_num_bases_processed(0),
          m_num_samples_processed(0),
          m_num_reads_processed(0),
          m_initialization_time(std::chrono::system_clock::now()),
          m_num_reads_expected(num_reads),
          m_active_threads(0) {
#ifdef _WIN32
    m_isatty = true;
#else
    m_isatty = isatty(fileno(stderr));
#endif

    if (m_num_reads_expected <= 100) {
        m_progress_bar_increment = 100;
    } else {
        m_progress_bar_increment = m_num_reads_expected / 100;
    }

    // Write to stdout for now.
    m_file = hts_open("-", "w");

    add_header();
    for (size_t i = 0; i < num_worker_threads; i++) {
        m_workers.push_back(
                std::make_unique<std::thread>(std::thread(&WriterNode::worker_thread, this)));
    }
}

WriterNode::~WriterNode() {
    terminate();
    for (auto& m : m_workers) {
        m->join();
    }
    if (m_header) {
        sam_hdr_destroy(m_header);
    }
    if (m_file) {
        hts_close(m_file);
    }
}

}  // namespace dorado
