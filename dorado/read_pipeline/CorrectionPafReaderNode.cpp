#include "CorrectionPafReaderNode.h"

#include "ClientInfo.h"
#include "ReadPipeline.h"
#include "utils/alignment_utils.h"
#include "utils/paf_utils.h"
#include "utils/timer_high_res.h"

#include <spdlog/spdlog.h>

#include <fstream>
#include <vector>

namespace dorado {

void CorrectionPafReaderNode::process(Pipeline& pipeline) {
    timer::TimerHighRes timer;

    std::ifstream file(m_paf_file);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open PAF file " + m_paf_file);
    }

    CorrectionAlignments alignments;

    size_t count_records = 0;
    std::string line;
    while (std::getline(file, line)) {
        utils::PafEntry entry = utils::parse_paf(line);

        // Skip all blacklisted targets.
        if (m_skip_set.count(entry.tname) > 0) {
            continue;
        }

        if (alignments.read_name != entry.tname) {
            if (m_reads_to_infer) {
                spdlog::trace(
                        "Pushed {} alignments for correction for "
                        "target {}. Number of alignment piles pushed until now: {}.",
                        std::size(alignments.qnames), alignments.read_name, m_reads_to_infer);
                pipeline.push_message(std::move(alignments));
            }
            alignments = CorrectionAlignments{};
            alignments.read_name = std::move(entry.tname);
            ++m_reads_to_infer;
        }

        utils::Overlap ovlp;
        ovlp.qstart = entry.qstart;
        ovlp.qend = entry.qend;
        ovlp.qlen = entry.qlen;
        ovlp.fwd = entry.strand == '+';
        ovlp.tstart = entry.tstart;
        ovlp.tend = entry.tend;
        ovlp.tlen = entry.tlen;

        alignments.qnames.push_back(std::move(entry.qname));
        alignments.overlaps.push_back(ovlp);

        const std::string_view cigar_str = utils::paf_aux_get(entry, "cg", 'Z');
        std::vector<CigarOp> cigar = parse_cigar_from_string(cigar_str);
        alignments.cigars.push_back(std::move(cigar));

        ++count_records;
        if ((count_records % 1000000) == 0) {
            spdlog::debug(
                    "Parsed {} PAF records in {} alignment piles. "
                    "Time: {:.2f} s",
                    count_records, m_reads_to_infer, timer.GetElapsedMilliseconds() / 1000.0f);
        }
    }
    if (!std::empty(alignments.qnames)) {
        spdlog::trace(
                "Final pushed {} alignments for correction for "
                "target {}. Number of piles pushed until now: {}.",
                std::size(alignments.qnames), alignments.read_name, m_reads_to_infer);
        pipeline.push_message(std::move(alignments));
    }

    spdlog::debug("PAF reading done in: {:.2f} s", timer.GetElapsedMilliseconds() / 1000.0f);
}

CorrectionPafReaderNode::CorrectionPafReaderNode(const std::string_view paf_file,
                                                 std::unordered_set<std::string> skip_set)
        : MessageSink(1, 1),
          m_paf_file(paf_file),
          m_reads_to_infer{0},
          m_skip_set{std::move(skip_set)} {}

stats::NamedStats CorrectionPafReaderNode::sample_stats() const {
    stats::NamedStats stats = stats::from_obj(m_work_queue);
    stats["num_reads_to_infer"] = static_cast<double>(m_reads_to_infer);
    return stats;
}

}  // namespace dorado
