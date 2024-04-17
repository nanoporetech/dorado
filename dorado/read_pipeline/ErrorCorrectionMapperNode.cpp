#include "ErrorCorrectionMapperNode.h"

#include "ClientInfo.h"
#include "alignment/Minimap2Aligner.h"
#include "alignment/Minimap2Index.h"
#include "alignment/Minimap2IndexSupportTypes.h"
#include "alignment/Minimap2Options.h"
#include "utils/PostCondition.h"
#include "utils/bam_utils.h"

#include <htslib/faidx.h>
#include <htslib/sam.h>
#include <minimap.h>
#include <spdlog/spdlog.h>

#include <cassert>
#include <filesystem>
#include <string>
#include <vector>

namespace {

[[maybe_unused]] std::vector<dorado::CigarOp> parse_cigar(const uint32_t* cigar,
                                                          uint32_t n_cigar,
                                                          bool fwd) {
    std::vector<dorado::CigarOp> cigar_ops;
    cigar_ops.reserve(n_cigar);
    for (uint32_t i = 0; i < n_cigar; i++) {
        uint32_t op = cigar[i] & 0xf;
        uint32_t len = cigar[i] >> 4;
        if (op == MM_CIGAR_MATCH) {
            cigar_ops.push_back({dorado::CigarOpType::MATCH, len});
        } else if (op == MM_CIGAR_INS) {
            cigar_ops.push_back({dorado::CigarOpType::DEL, len});
        } else if (op == MM_CIGAR_DEL) {
            cigar_ops.push_back({dorado::CigarOpType::INS, len});
        } else {
            throw std::runtime_error("Unknown cigar op: " + std::to_string(op));
        }
    }
    if (!fwd) {
        std::reverse(cigar_ops.begin(), cigar_ops.end());
    }
    return cigar_ops;
}

}  // namespace

namespace dorado {

ErrorCorrectionMapperNode::ErrorCorrectionMapperNode(const std::string& index_file, int threads)
        : MessageSink(10000, threads), m_index_file(index_file) {
    alignment::Minimap2Options options = alignment::dflt_options;
    options.kmer_size = 25;
    options.window_size = 17;
    options.mm2_preset = "ava-ont";
    options.bandwidth = 150;
    options.min_chain_score = 4000;
    options.zdrop = options.zdrop_inv = 200;
    options.occ_dist = 200;
    options.cs = "short";
    options.dual = "yes";

    m_index = std::make_shared<alignment::Minimap2Index>();
    if (!m_index->initialise(options)) {
        spdlog::error("Failed to initialize with options.");
        throw std::runtime_error("");
    } else {
        spdlog::info("Initialized index options");
    }
    spdlog::info("Loading index...");
    if (m_index->load(index_file, threads) != alignment::IndexLoadResult::success) {
        spdlog::error("Failed to load index file {}", index_file);
        throw std::runtime_error("");
    } else {
        spdlog::info("Loaded mm2 index.");
    }
    // Create aligner.
    m_aligner = std::make_unique<alignment::Minimap2Aligner>(m_index);
    // Create fastx index.
    spdlog::info("Creating input fastq index...");
    char* idx_name = fai_path(index_file.c_str());
    spdlog::info("Looking for idx {}", idx_name);
    if (!std::filesystem::exists(idx_name)) {
        if (fai_build(index_file.c_str()) != 0) {
            spdlog::error("Failed to build index for file {}", index_file);
            throw std::runtime_error("");
        }
        spdlog::info("Created fastq index.");
    }
    free(idx_name);
    start_input_processing(&ErrorCorrectionMapperNode::input_thread_fn, this);
}

void ErrorCorrectionMapperNode::input_thread_fn() {
    Message message;
    mm_tbuf_t* tbuf = mm_tbuf_init();
    auto fastx_reader = std::make_unique<hts_io::FastxRandomReader>(m_index_file);
    while (get_input_message(message)) {
        if (std::holds_alternative<BamPtr>(message)) {
            auto read = std::get<BamPtr>(std::move(message));
            const std::string read_name = bam_get_qname(read.get());
            const std::string read_seq = utils::extract_sequence(read.get());
            auto start = std::chrono::high_resolution_clock::now();
            auto [reg, hits] = m_aligner->get_mapping(read.get(), tbuf);
            auto clear_alignments = utils::PostCondition([reg, hits]() {
                for (int j = 0; j < hits; j++) {
                    free(reg[j].p);
                }
                free(reg);
            });
            (void)reg;
            (void)hits;
            auto end = std::chrono::high_resolution_clock::now();
            {
                std::chrono::duration<double> duration = end - start;
                std::lock_guard<std::mutex> lock(mm2mutex);
                mm2Duration += duration;
            }
            auto alignments = extract_alignments(reg, hits, fastx_reader.get(), read_seq);
            //CorrectionAlignments alignments;
            alignments.read_name = std::move(read_name);
            alignments.read_seq = std::move(read_seq);
            alignments.read_qual = utils::extract_quality(read.get());
            send_message_to_sink(std::move(alignments));
        } else {
            send_message_to_sink(std::move(message));
            continue;
        }
    }
    mm_tbuf_destroy(tbuf);
}

void ErrorCorrectionMapperNode::terminate(const FlushOptions&) {
    stop_input_processing();
    spdlog::info("time for mm2 {}", mm2Duration.count());
    spdlog::info("time for fastq read {}", fastqDuration.count());
}

CorrectionAlignments ErrorCorrectionMapperNode::extract_alignments(
        const mm_reg1_t* reg,
        int hits,
        const hts_io::FastxRandomReader* reader,
        const std::string& qread) {
    (void)reader;
    ;
    (void)qread;
    dorado::CorrectionAlignments alignments;
    for (int j = 0; j < hits; j++) {
        Overlap ovlp;

        // mapping region
        auto aln = &reg[j];

        ovlp.qid = alignments.seqs.size();
        ovlp.qstart = aln->rs;
        ovlp.qend = aln->re;
        ovlp.fwd = !aln->rev;
        ovlp.tstart = aln->qs;
        ovlp.tend = aln->qe;

        const std::string qname(m_index->index()->seq[aln->rid].name);

        //if (qname != "e3066d3e-2bdf-4803-89b9-0f077ac7ff7f")
        //    continue;
        auto start = std::chrono::high_resolution_clock::now();
        alignments.seqs.push_back(reader->fetch_seq(qname));
        alignments.quals.push_back(reader->fetch_qual(qname));
        auto end = std::chrono::high_resolution_clock::now();
        {
            std::chrono::duration<double> duration = end - start;
            //std::lock_guard<std::mutex> lock(fastqmutex);
            fastqDuration += duration;
        }
        alignments.qnames.push_back(qname);

        ovlp.qlen = (int)alignments.seqs.back().length();
        ovlp.tlen = (int)qread.length();

        size_t n_cigar = aln->p ? aln->p->n_cigar : 0;
        alignments.cigars.push_back(parse_cigar(aln->p->cigar, n_cigar, ovlp.fwd));

        alignments.overlaps.push_back(std::move(ovlp));
    }
    return alignments;
}

alignment::HeaderSequenceRecords ErrorCorrectionMapperNode::get_sequence_records_for_header()
        const {
    return m_aligner->get_sequence_records_for_header();
}

stats::NamedStats ErrorCorrectionMapperNode::sample_stats() const {
    return stats::from_obj(m_work_queue);
}

}  // namespace dorado
