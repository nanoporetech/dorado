#include "ErrorCorrectionMapperNode.h"

#include "ClientInfo.h"
#include "HtsReader.h"
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

[[maybe_unused]] std::vector<dorado::CigarOp> parse_cigar(const uint32_t* cigar, uint32_t n_cigar) {
    std::vector<dorado::CigarOp> cigar_ops;
    cigar_ops.reserve(n_cigar);
    for (uint32_t i = 0; i < n_cigar; i++) {
        uint32_t op = cigar[i] & 0xf;
        uint32_t len = cigar[i] >> 4;
        if (op == MM_CIGAR_MATCH) {
            cigar_ops.push_back({dorado::CigarOpType::MATCH, len});
        } else if (op == MM_CIGAR_INS) {
            cigar_ops.push_back({dorado::CigarOpType::INS, len});
        } else if (op == MM_CIGAR_DEL) {
            cigar_ops.push_back({dorado::CigarOpType::DEL, len});
        } else {
            throw std::runtime_error("Unknown cigar op: " + std::to_string(op));
        }
    }
    return cigar_ops;
}

}  // namespace

namespace dorado {

void ErrorCorrectionMapperNode::extract_alignments(const mm_reg1_t* reg,
                                                   int hits,
                                                   const std::string& qread,
                                                   const std::string& qname) {
    for (int j = 0; j < hits; j++) {
        // mapping region
        auto aln = &reg[j];
        const std::string tname(m_index->index()->seq[aln->rid].name);

        std::unique_lock<std::mutex> lock(m_correction_mtx);
        if (m_read_mutex.find(tname) == m_read_mutex.end()) {
            m_read_mutex.emplace(tname, std::make_unique<std::mutex>());
            m_correction_records.emplace(tname, CorrectionAlignments{});
        }
        auto& mtx = *m_read_mutex[tname];
        lock.unlock();

        Overlap ovlp;
        ovlp.qstart = aln->qs;
        ovlp.qend = aln->qe;
        ovlp.fwd = !aln->rev;
        ovlp.tstart = aln->rs;
        ovlp.tend = aln->re;
        ovlp.qlen = (int)qread.length();

        uint32_t n_cigar = aln->p ? aln->p->n_cigar : 0;
        auto cigar = parse_cigar(aln->p->cigar, n_cigar);

        std::lock_guard<std::mutex> aln_lock(mtx);
        auto& alignments = m_correction_records[tname];
        if (alignments.read_name.empty()) {
            alignments.read_name = tname;
        }

        ovlp.qid = (int)alignments.seqs.size();

        alignments.qnames.push_back(qname);

        alignments.cigars.push_back(std::move(cigar));
        alignments.overlaps.push_back(std::move(ovlp));
    }
}

void ErrorCorrectionMapperNode::input_thread_fn() {
    BamPtr read;
    MmTbufPtr tbuf(mm_tbuf_init());
    while (m_reads_queue.try_pop(read) != utils::AsyncQueueStatus::Terminate) {
        const std::string read_name = bam_get_qname(read.get());
        const std::string read_seq = utils::extract_sequence(read.get());
        std::tuple<mm_reg1_t*, int> mapping = m_aligner->get_mapping(read.get(), tbuf.get());
        mm_reg1_t* reg = std::get<0>(mapping);
        int hits = std::get<1>(mapping);

        auto clear_alignments = utils::PostCondition([reg, hits]() {
            for (int j = 0; j < hits; j++) {
                free(reg[j].p);
            }
            free(reg);
        });
        extract_alignments(reg, hits, read_seq, read_name);
        m_alignments_processed++;
        // TODO: Remove and move to ProgressTracker
        if (m_alignments_processed.load() % 10000 == 0) {
            spdlog::debug("Alignments processed {}", m_alignments_processed.load());
        }
    }
}

void ErrorCorrectionMapperNode::load_read_fn() {
    m_reads_queue.restart();
    HtsReader reader(m_index_file, {});
    while (reader.read()) {
        m_reads_queue.try_push(BamPtr(bam_dup1(reader.record.get())));
        m_reads_read++;
        // TODO: Remove and move to ProgressTracker
        if (m_reads_read.load() % 10000 == 0) {
            spdlog::debug("Read {} reads", m_reads_read.load());
        }
    }
    m_reads_queue.terminate();
}

void ErrorCorrectionMapperNode::send_data_fn(Pipeline& pipeline) {
    while (!m_copy_terminate.load()) {
        std::unique_lock<std::mutex> lock(m_copy_mtx);
        m_copy_cv.wait(lock, [&] {
            return (!m_shadow_correction_records.empty() || m_copy_terminate.load());
        });

        spdlog::debug("Pushing {} records for correction", m_shadow_correction_records.size());
        for (auto& [n, r] : m_shadow_correction_records) {
            pipeline.push_message(std::move(r));
        }
        m_shadow_correction_records.clear();
    }
}

void ErrorCorrectionMapperNode::process(Pipeline& pipeline) {
    std::thread reader_thread;
    std::vector<std::thread> aligner_threads;
    std::thread copy_thread =
            std::thread(&ErrorCorrectionMapperNode::send_data_fn, this, std::ref(pipeline));
    int index = 0;
    do {
        spdlog::debug("Align with index {}", index);
        m_reads_read.store(0);
        m_alignments_processed.store(0);

        // Create aligner.
        m_aligner = std::make_unique<alignment::Minimap2Aligner>(m_index);
        // 1. Start threads for aligning reads.
        reader_thread = std::thread(&ErrorCorrectionMapperNode::load_read_fn, this);
        // 2. Start thread for generating reads.
        for (int i = 0; i < m_num_threads; i++) {
            aligner_threads.push_back(
                    std::thread(&ErrorCorrectionMapperNode::input_thread_fn, this));
        }
        // 3. Wait for alignments to finish and all reads to be read
        if (reader_thread.joinable()) {
            reader_thread.join();
        }
        //reader_thread.reset();
        for (auto& t : aligner_threads) {
            if (t.joinable()) {
                t.join();
            }
        }
        aligner_threads.clear();
        {
            // Only copy when the thread sending alignments to downstream pipeline
            // is done.
            std::unique_lock<std::mutex> lock(m_copy_mtx);
            m_shadow_correction_records = std::move(m_correction_records);
        }
        m_copy_cv.notify_one();

        m_correction_records.clear();
        m_read_mutex.clear();
        // 4. Load next index and loop
        index++;
    } while (m_index->load_next_chunk(m_num_threads) != alignment::IndexLoadResult::end_of_index);

    m_copy_terminate.store(true);
    if (copy_thread.joinable()) {
        copy_thread.join();
    }
    //copy_thread.reset();
}

ErrorCorrectionMapperNode::ErrorCorrectionMapperNode(const std::string& index_file, int threads)
        : MessageSink(10000, threads),
          m_index_file(index_file),
          m_num_threads(threads),
          m_reads_queue(5000) {
    alignment::Minimap2Options options = alignment::dflt_options;
    options.kmer_size = 25;
    options.window_size = 17;
    options.index_batch_size = 800000000ull;
    options.mm2_preset = "ava-ont";
    options.bandwidth = 150;
    options.bandwidth_long = 2000;
    options.min_chain_score = 4000;
    options.zdrop = options.zdrop_inv = 200;
    options.occ_dist = 200;
    options.cs = "short";
    options.dual = "yes";

    m_index = std::make_shared<alignment::Minimap2Index>();
    if (!m_index->initialise(options)) {
        throw std::runtime_error("Failed to initialize with options.");
    } else {
        spdlog::debug("Initialized index options.");
    }
    spdlog::debug("Loading index...");
    if (m_index->load(index_file, threads, true) != alignment::IndexLoadResult::success) {
        throw std::runtime_error("Failed to load index file " + index_file);
    } else {
        spdlog::debug("Loaded mm2 index.");
    }
}

stats::NamedStats ErrorCorrectionMapperNode::sample_stats() const {
    return stats::from_obj(m_work_queue);
}

}  // namespace dorado
