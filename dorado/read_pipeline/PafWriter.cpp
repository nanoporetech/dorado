#include "PafWriter.h"

#include "read_pipeline/ReadPipeline.h"
#include "utils/alignment_utils.h"
#include "utils/paf_utils.h"
#include "utils/sequence_utils.h"

#include <minimap.h>

#include <cassert>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace dorado {

std::vector<dorado::CigarOp> parse_cigar(const uint32_t* cigar, uint32_t n_cigar) {
    std::vector<dorado::CigarOp> cigar_ops;
    cigar_ops.resize(n_cigar);
    for (uint32_t i = 0; i < n_cigar; i++) {
        uint32_t op = cigar[i] & 0xf;
        uint32_t len = cigar[i] >> 4;
        if (op == MM_CIGAR_MATCH) {
            cigar_ops[i] = {dorado::CigarOpType::MATCH, len};
        } else if (op == MM_CIGAR_INS) {
            cigar_ops[i] = {dorado::CigarOpType::INS, len};
        } else if (op == MM_CIGAR_DEL) {
            cigar_ops[i] = {dorado::CigarOpType::DEL, len};
        } else {
            throw std::runtime_error("Unknown cigar op: " + std::to_string(op));
        }
    }
    return cigar_ops;
}

PafWriter::PafWriter() : MessageSink(10000, 1) {
    start_input_processing(&PafWriter::input_thread_fn, this);
}

PafWriter::~PafWriter() { stop_input_processing(); }

void PafWriter::input_thread_fn() {
    Message message;
    while (get_input_message(message)) {
        if (!std::holds_alternative<CorrectionAlignments>(message)) {
            continue;
        }

        auto alignments = std::get<CorrectionAlignments>(std::move(message));

        for (size_t i = 0; i < alignments.qnames.size(); i++) {
            utils::PafEntry entry;

            entry.tname = alignments.read_name;
            entry.tstart = alignments.overlaps[i].tstart;
            entry.tend = alignments.overlaps[i].tend;
            entry.tlen = alignments.overlaps[i].tlen;
            entry.qstart = alignments.overlaps[i].qstart;
            entry.qend = alignments.overlaps[i].qend;
            entry.qlen = alignments.overlaps[i].qlen;
            entry.qname = alignments.qnames[i];
            entry.strand = alignments.overlaps[i].fwd ? '+' : '-';
            auto cig = parse_cigar(alignments.mm2_cigars[i].data(),
                                   (uint32_t)alignments.mm2_cigars[i].size());
            entry.add_aux_tag("cg", 'Z', utils::serialize_cigar(cig));
            std::cout << utils::serialize_paf(entry) << std::endl;
        }
    }
}

stats::NamedStats PafWriter::sample_stats() const {
    auto stats = stats::from_obj(m_work_queue);
    return stats;
}

void PafWriter::terminate(const FlushOptions&) { stop_input_processing(); }

}  // namespace dorado
