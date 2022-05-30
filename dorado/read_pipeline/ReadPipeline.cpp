#include "ReadPipeline.h"

#include "utils/sequence_utils.h"

#include <chrono>

using namespace std::chrono_literals;

std::vector<std::string> Read::generate_read_tags() const {
    // GCC doesn't support <format> yet...
    std::vector<std::string> tags = {"qs:i:" + std::to_string(static_cast<int>(std::round(
                                                       utils::mean_qscore_from_qstring(qstring)))),
                                     "ns:i:" + std::to_string(num_samples),
                                     "ts:i:" + std::to_string(num_trimmed_samples),
                                     "mx:i:" + std::to_string(attributes.mux),
                                     "ch:i:" + std::to_string(attributes.channel_number),
                                     "st:Z:" + attributes.start_time,
                                     "rn:i:" + std::to_string(attributes.read_number),
                                     "f5:Z:" + attributes.fast5_filename};

    return tags;
}

std::vector<std::string> Read::extract_sam_lines() const {
    if (read_id.empty()) {
        throw std::runtime_error("Empty read_name string provided");
    }
    if (seq.size() != qstring.size()) {
        throw std::runtime_error("Sequence and qscore do not match size for read id " + read_id);
    }
    if (seq.empty()) {
        throw std::runtime_error("Empty sequence and qstring provided for read id " + read_id);
    }

    std::ostringstream read_tags_stream;
    auto read_tags = generate_read_tags();
    for (const auto& tag : read_tags) {
        read_tags_stream << "\t" << tag;
    }

    std::vector<std::string> sam_lines;
    if (mappings.empty()) {
        uint32_t flags = 4;              // 4 = UNMAPPED
        std::string ref_seq = "*";       // UNMAPPED
        int leftmost_pos = -1;           // UNMAPPED - will be written as 0
        int map_q = 0;                   // UNMAPPED
        std::string cigar_string = "*";  // UNMAPPED
        std::string r_next = "*";
        int next_pos = -1;  // UNMAPPED - will be written as 0
        size_t template_length = seq.size();

        std::ostringstream sam_line;
        sam_line << read_id << "\t"             // QNAME
                 << flags << "\t"               // FLAG
                 << ref_seq << "\t"             // RNAME
                 << (leftmost_pos + 1) << "\t"  // POS
                 << map_q << "\t"               // MAPQ
                 << cigar_string << "\t"        // CIGAR
                 << r_next << "\t"              // RNEXT
                 << (next_pos + 1) << "\t"      // PNEXT
                 << (template_length) << "\t"   // TLEN
                 << seq << "\t"                 // SEQ
                 << qstring;                    // QUAL

        sam_line << read_tags_stream.str();
        sam_lines.push_back(sam_line.str());
    }

    for (const auto& mapping : mappings) {
        throw std::runtime_error("Mapped alignments not yet implemented");
    }

    return sam_lines;
}

void ReadSink::push_read(std::shared_ptr<Read>& read) {
    std::unique_lock<std::mutex> push_read_cv_lock(m_push_read_cv_mutex);
    while (!m_push_read_cv.wait_for(push_read_cv_lock, 100ms,
                                    [this] { return m_reads.size() < m_max_reads; })) {
    }
    {
        std::unique_lock<std::mutex> lock(m_cv_mutex);
        m_reads.push_back(read);
    }
    m_cv.notify_one();
}

ReadSink::ReadSink(size_t max_reads) : m_max_reads(max_reads) {}
