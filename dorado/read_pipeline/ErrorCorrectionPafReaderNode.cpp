#include "ErrorCorrectionPafReaderNode.h"

#include "ClientInfo.h"
#include "utils/alignment_utils.h"
#include "utils/paf_utils.h"

#include <spdlog/spdlog.h>

#include <cassert>
#include <filesystem>
#include <string>
#include <vector>

namespace dorado {

void ErrorCorrectionPafReaderNode::process(Pipeline& pipeline) {
    std::ifstream file(m_paf_file);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open PAF file " + m_paf_file);
    }

    bool start = true;
    CorrectionAlignments alignments;

    size_t count = 0;
    std::string line;
    while (std::getline(file, line)) {
        auto entry = utils::parse_paf(line);

        if (alignments.read_name != entry.tname) {
            if (!start) {
                spdlog::trace("Pushed alignment for {}", alignments.read_name);
                pipeline.push_message(std::move(alignments));
            }
            alignments = CorrectionAlignments{};
            alignments.read_name = entry.tname;
            start = false;
        }

        Overlap ovlp;
        ovlp.qstart = entry.qstart;
        ovlp.qend = entry.qend;
        ovlp.qlen = entry.qlen;
        ovlp.fwd = entry.strand == '+';
        ovlp.tstart = entry.tstart;
        ovlp.tend = entry.tend;
        ovlp.tlen = entry.tlen;

        alignments.qnames.push_back(entry.qname);
        alignments.overlaps.push_back(ovlp);

        auto cigar = utils::parse_cigar(utils::paf_aux_get(entry, "cg", 'Z'));
        alignments.cigars.push_back(std::move(cigar));
        count++;
        if (count % 1000000 == 0) {
            spdlog::debug("Parsed {} PAF rows", count);
        }
    }
    pipeline.push_message(std::move(alignments));

    spdlog::debug("Pushing {} records for correction", m_correction_records.size());
}

ErrorCorrectionPafReaderNode::ErrorCorrectionPafReaderNode(const std::string& paf_file)
        : MessageSink(1, 1), m_paf_file(paf_file) {}

stats::NamedStats ErrorCorrectionPafReaderNode::sample_stats() const {
    stats::NamedStats stats = stats::from_obj(m_work_queue);
    stats["num_reads_to_infer"] = static_cast<double>(m_reads_to_correct);
    return stats;
}

}  // namespace dorado
