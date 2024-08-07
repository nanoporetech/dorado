#include "ErrorCorrectionMapperNode.h"

#include "ClientInfo.h"
#include "HtsReader.h"
#include "alignment/Minimap2Aligner.h"
#include "alignment/Minimap2Index.h"
#include "alignment/Minimap2IndexSupportTypes.h"
#include "alignment/Minimap2Options.h"
#include "alignment/minimap2_args.h"
#include "alignment/minimap2_wrappers.h"
#include "utils/PostCondition.h"
#include "utils/alignment_utils.h"
#include "utils/bam_utils.h"
#include "utils/thread_naming.h"

#include <htslib/faidx.h>
#include <htslib/sam.h>
#include <minimap.h>
#include <spdlog/spdlog.h>

#include <cassert>
#include <filesystem>
#include <string>
#include <vector>

namespace dorado {

void ErrorCorrectionMapperNode::extract_alignments(const mm_reg1_t* reg,
                                                   int hits,
                                                   const std::string& qread,
                                                   const std::string& qname) {
    for (int j = 0; j < hits; j++) {
        // mapping region
        auto aln = &reg[j];

        if (aln->p == 0) {
            continue;
        }

        const auto& ref = m_index->index()->seq[aln->rid];
        const std::string tname(ref.name);

        // Skip self alignment.
        if (qname == tname) {
            continue;
        }

        std::unique_lock<std::mutex> lock(m_correction_mtx);
        if (m_read_mutex.find(tname) == m_read_mutex.end()) {
            m_read_mutex.emplace(tname, std::make_unique<std::mutex>());
            CorrectionAlignments new_aln;
            new_aln.read_name = tname;
            m_correction_records.emplace(tname, std::move(new_aln));
            m_processed_queries_per_target.emplace(tname, std::unordered_set<std::string>());
        }
        auto& mtx = *m_read_mutex[tname];
        lock.unlock();

        {
            std::lock_guard<std::mutex> aln_lock(mtx);
            auto& processed_queries = m_processed_queries_per_target[tname];
            if (processed_queries.find(qname) != processed_queries.end()) {
                // Query/target pair has been processed before. Assume that
                // the first one processed is the best one, and ignore
                // the rest.
                continue;
            } else {
                processed_queries.insert(qname);
            }
        }

        Overlap ovlp;
        ovlp.qstart = aln->qs;
        ovlp.qend = aln->qe;
        ovlp.qlen = (int)qread.length();
        ovlp.fwd = !aln->rev;
        ovlp.tstart = aln->rs;
        ovlp.tend = aln->re;
        ovlp.tlen = ref.len;

        if (ovlp.qlen < ovlp.qstart || ovlp.qlen < ovlp.qend) {
            spdlog::warn(
                    "Inconsistent query alignment detected: tname {} tlen {} tstart {} tend {} "
                    "qname {} qlen {} qstart {} qend {}",
                    tname, ovlp.tlen, ovlp.tstart, ovlp.tend, qname, ovlp.qlen, ovlp.qstart,
                    ovlp.qend);
            continue;
        }

        if (ovlp.tlen < ovlp.tstart || ovlp.tlen < ovlp.tend) {
            spdlog::warn(
                    "Inconsistent target alignment detected: tname {} tlen {} tstart {} tend {} "
                    "qname {} qlen {} qstart {} qend {}",
                    tname, ovlp.tlen, ovlp.tstart, ovlp.tend, qname, ovlp.qlen, ovlp.qstart,
                    ovlp.qend);
            continue;
        }

        // Non-const because it will be moved below.
        std::vector<CigarOp> cigar = convert_mm2_cigar(aln->p->cigar, aln->p->n_cigar);

        {
            std::lock_guard<std::mutex> aln_lock(mtx);

            auto& alignments = m_correction_records[tname];
            alignments.qnames.push_back(qname);
            alignments.cigars.push_back(std::move(cigar));
            alignments.overlaps.push_back(std::move(ovlp));
        }
    }
}

void ErrorCorrectionMapperNode::input_thread_fn() {
    utils::set_thread_name("errcorr_node");
    BamPtr read;
    MmTbufPtr tbuf(mm_tbuf_init());
    while (m_reads_queue.try_pop(read) != utils::AsyncQueueStatus::Terminate) {
        const std::string read_name = bam_get_qname(read.get());
        const std::string read_seq = utils::extract_sequence(read.get());
        std::tuple<mm_reg1_t*, int> mapping = m_aligner->get_mapping(read.get(), tbuf.get());
        mm_reg1_t* reg = std::get<0>(mapping);
        int hits = std::get<1>(mapping);
        extract_alignments(reg, hits, read_seq, read_name);
        m_alignments_processed++;
        // TODO: Remove and move to ProgressTracker
        if (m_alignments_processed.load() % 10000 == 0) {
            std::unique_lock<std::mutex> lock(m_correction_mtx);
            size_t total = 0;
            for (auto& [_, r] : m_correction_records) {
                total += r.size();
            }
            spdlog::debug("Alignments processed {}, total m_corrected_records size {} MB",
                          m_alignments_processed.load(), (float)total / (1024 * 1024));
        }

        for (int j = 0; j < hits; j++) {
            free(reg[j].p);
        }
        free(reg);
    }
}

void ErrorCorrectionMapperNode::load_read_fn() {
    utils::set_thread_name("errcorr_load");
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
    utils::set_thread_name("errcorr_copy");
    while (!m_copy_terminate.load()) {
        std::unique_lock<std::mutex> lock(m_copy_mtx);
        m_copy_cv.wait(lock, [&] {
            return (!m_shadow_correction_records.empty() || m_copy_terminate.load());
        });

        spdlog::debug("Pushing {} records for correction", m_shadow_correction_records.size());
        m_reads_to_infer.store(m_reads_to_infer.load() + m_shadow_correction_records.size());
        for (auto& [_, r] : m_shadow_correction_records) {
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
    do {
        spdlog::debug("Align with index {}", m_current_index);
        m_reads_read.store(0);
        m_alignments_processed.store(0);

        // Create aligner.
        m_aligner = std::make_unique<alignment::Minimap2Aligner>(m_index);
        // 1. Start thread for generating reads.
        reader_thread = std::thread(&ErrorCorrectionMapperNode::load_read_fn, this);
        // 2. Start threads for aligning reads.
        for (int i = 0; i < m_num_threads; i++) {
            aligner_threads.push_back(
                    std::thread(&ErrorCorrectionMapperNode::input_thread_fn, this));
        }
        // 3. Wait for alignments to finish and all reads to be read
        if (reader_thread.joinable()) {
            reader_thread.join();
        }
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
        m_processed_queries_per_target.clear();
        // 4. Load next index and loop
        m_current_index++;
    } while (m_index->load_next_chunk(m_num_threads) != alignment::IndexLoadResult::end_of_index);

    m_copy_terminate.store(true);
    m_copy_cv.notify_all();
    if (copy_thread.joinable()) {
        copy_thread.join();
    }
}

ErrorCorrectionMapperNode::ErrorCorrectionMapperNode(const std::string& index_file,
                                                     int threads,
                                                     uint64_t index_size)
        : MessageSink(10000, threads),
          m_index_file(index_file),
          m_num_threads(threads),
          m_reads_queue(5000) {
    auto options = alignment::create_preset_options("ava-ont");
    auto& index_options = options.index_options->get();
    index_options.k = 25;
    index_options.w = 17;
    index_options.batch_size = index_size;
    auto& mapping_options = options.mapping_options->get();
    mapping_options.bw = 150;
    mapping_options.bw_long = 2000;
    mapping_options.min_chain_score = 4000;
    mapping_options.zdrop = 200;
    mapping_options.zdrop_inv = 200;
    mapping_options.occ_dist = 200;
    mapping_options.flag |= MM_F_EQX;

    // --cs short
    alignment::mm2::apply_cs_option(options, "short");

    // --dual yes
    alignment::mm2::apply_dual_option(options, "yes");

    // reset to larger minimap2 defaults
    mm_mapopt_t minimap_default_mapopt;
    mm_mapopt_init(&minimap_default_mapopt);
    mapping_options.cap_kalloc = minimap_default_mapopt.cap_kalloc;
    mapping_options.max_sw_mat = minimap_default_mapopt.max_sw_mat;

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
    m_index_seqs = m_index->index()->n_seq;
}

stats::NamedStats ErrorCorrectionMapperNode::sample_stats() const {
    stats::NamedStats stats = stats::from_obj(m_work_queue);
    stats["num_reads_aligned"] = m_alignments_processed.load();
    stats["num_reads_to_infer"] = static_cast<double>(m_reads_to_infer.load());
    stats["index_seqs"] = m_index_seqs;
    stats["current_idx"] = m_current_index;
    return stats;
}

}  // namespace dorado
