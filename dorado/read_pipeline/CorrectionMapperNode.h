#pragma once

#include "MessageSink.h"
#include "alignment/Minimap2Aligner.h"
#include "alignment/Minimap2Index.h"
#include "alignment/Minimap2IndexSupportTypes.h"
#include "messages.h"
#include "utils/AsyncQueue.h"
#include "utils/stats.h"
#include "utils/types.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dorado {

class Pipeline;

class CorrectionMapperNode : public MessageSink {
public:
    CorrectionMapperNode(const std::string& index_file,
                         int32_t threads,
                         uint64_t index_size,
                         std::string furthest_skip_header,
                         std::unordered_set<std::string> skip_set,
                         int32_t run_block_id,
                         int32_t kmer_size,
                         int32_t window_size,
                         int32_t min_chain_score,
                         float mid_occ_frac);

    ~CorrectionMapperNode() = default;
    std::string get_name() const override { return "CorrectionMapperNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions&) override {};
    void restart() override {}
    // Main driver function.
    void process(Pipeline& pipeline);

    bool load_next_index_block();

    int get_current_index_block_id() const;
    int get_index_seqs() const;

private:
    std::string m_index_file;
    int m_num_threads;

    std::unique_ptr<alignment::Minimap2Aligner> m_aligner;
    std::shared_ptr<alignment::Minimap2Index> m_index;

    void input_thread_fn();
    void load_read_fn();
    void send_data_fn(Pipeline& pipeline);

    void extract_alignments(const mm_reg1_t* reg,
                            int hits,
                            const std::string& qread,
                            const std::string& qname);

    // Queue for reads being aligned.
    utils::AsyncQueue<BamPtr> m_reads_queue;

    // Map to collects alignments by target id.
    std::mutex m_correction_mtx;
    std::unordered_map<std::string, CorrectionAlignments> m_correction_records;
    std::unordered_map<std::string, std::unordered_set<std::string>> m_processed_queries_per_target;

    std::mutex m_copy_mtx;
    std::condition_variable m_copy_cv;
    std::vector<std::unordered_map<std::string, CorrectionAlignments>> m_shadow_correction_records;

    // Mutex per target id to prevent a global lock across all targets.
    std::unordered_map<std::string, std::unique_ptr<std::mutex>> m_read_mutex;

    int m_index_seqs{0};
    int m_current_index{0};
    std::atomic<int> m_reads_read{0};
    std::atomic<int> m_alignments_processed{0};
    std::atomic<size_t> m_reads_to_infer{0};

    std::atomic<bool> m_copy_terminate{false};

    std::string m_furthest_skip_header;
    std::unordered_set<std::string> m_skip_set;
    int m_run_block_id{-1};
};

}  // namespace dorado
