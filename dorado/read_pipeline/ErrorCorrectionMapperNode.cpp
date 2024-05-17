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

const size_t MAX_OVERLAPS_PER_READ = 500;

namespace {

std::vector<uint32_t> copy_mm2_cigar(const uint32_t* cigar, uint32_t n_cigar) {
    std::vector<uint32_t> cigar_ops(cigar, cigar + n_cigar);
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

        uint32_t n_cigar = aln->p->n_cigar;
        auto cigar = copy_mm2_cigar(aln->p->cigar, n_cigar);

        {
            std::lock_guard<std::mutex> aln_lock(mtx);

            auto& alignments = m_correction_records[tname];

            // Cap total overlaps per read.
            if (alignments.qnames.size() >= MAX_OVERLAPS_PER_READ) {
                continue;
            }

            if (alignments.read_name.empty()) {
                alignments.read_name = tname;
            }

            alignments.qnames.push_back(qname);

            alignments.mm2_cigars.push_back(std::move(cigar));
            alignments.overlaps.push_back(std::move(ovlp));
        }
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
    int index = 0;
    do {
        spdlog::debug("Align with index {}", index);
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
        index++;
    } while (m_index->load_next_chunk(m_num_threads) != alignment::IndexLoadResult::end_of_index);

    m_copy_terminate.store(true);
    if (copy_thread.joinable()) {
        copy_thread.join();
    }
}

ErrorCorrectionMapperNode::ErrorCorrectionMapperNode(const std::string& index_file, int threads)
        : MessageSink(10000, threads),
          m_index_file(index_file),
          m_num_threads(threads),
          m_reads_queue(5000) {
    alignment::Minimap2Options options = alignment::dflt_options;
    options.kmer_size = 25;
    options.window_size = 17;
    options.index_batch_size = 8000000000ull;
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
