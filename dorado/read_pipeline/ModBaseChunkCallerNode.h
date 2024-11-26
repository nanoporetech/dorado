#pragma once

#include "modbase/ModBaseModelConfig.h"
#include "modbase/ModBaseRunner.h"
#include "read_pipeline/MessageSink.h"
#include "utils/AsyncQueue.h"
#include "utils/stats.h"

#include <spdlog/spdlog.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_set>
#include <vector>

namespace dorado {

class ModBaseChunkCallerNode : public MessageSink {
    struct WorkingRead;

    struct ModBaseChunk {
        ModBaseChunk(std::shared_ptr<WorkingRead> working_read_,
                     int model_id_,
                     int base_id_,
                     int64_t signal_start_,
                     int64_t hit_start_,
                     int64_t num_states_)
                : working_read(std::move(working_read_)),
                  model_id(model_id_),
                  base_id(base_id_),
                  signal_start(signal_start_),
                  hit_start(hit_start_),
                  num_states(num_states_) {}

        std::shared_ptr<WorkingRead> working_read;
        const int model_id;
        const int base_id;
        // The start index into the working read in the PADDED signal
        const int64_t signal_start;
        // The start index into the context hits
        const int64_t hit_start;
        // The number of states predicted by the modbase model `num_mods + 1`
        const int64_t num_states;
        // The model predictions for this chunk arranged in `[canonical, mod1, .., modN, canonical, mod1, ..]`
        std::vector<float> scores;
    };

public:
    ModBaseChunkCallerNode(std::vector<modbase::RunnerPtr> model_runners,
                           size_t num_threads,
                           size_t canonical_stride,
                           size_t max_reads);
    ~ModBaseChunkCallerNode();
    std::string get_name() const override { return "ModBaseChunkCallerNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions&) override { terminate_impl(); }
    void restart() override;

    static std::optional<int64_t> next_hit(const std::vector<int64_t>& ctx_hit_signal_idxs,
                                           const int64_t chunk_signal_start);

    static int64_t resolve_score_index(const int64_t hit_sig_abs,
                                       const int64_t chunk_signal_start,
                                       const int64_t scores_states,
                                       const int64_t chunk_size,
                                       const int64_t context_samples_before,
                                       const int64_t context_samples_after,
                                       const int64_t modbase_stride);

    static std::vector<std::pair<int64_t, int64_t>> get_chunk_starts(
            const int64_t signal_len,
            const std::vector<int64_t>& hits_to_sig,
            const int64_t chunk_size,
            const int64_t context_samples_before,
            const int64_t context_samples_after,
            const bool end_align_last_chunk);

private:
    using ModBaseChunks = std::vector<std::unique_ptr<ModBaseChunkCallerNode::ModBaseChunk>>;
    using PerBaseIntVec = std::array<std::vector<int64_t>, 4>;

    void start_threads();
    void terminate_impl();

    // Determine the modbase alphabet from all callers and calculate offset positions for the results
    void init_modbase_info();

    void input_thread_fn();

    void chunk_caller_thread_fn(size_t worker_id, size_t model_id);

    void output_thread_fn();

    void simplex_mod_call(Message&& message);

    // Called by chunk_caller_thread_fn, calls the model and enqueues the results
    void call_batch(size_t worker_id, size_t model_id, ModBaseChunks& batched_chunks);

    std::vector<modbase::RunnerPtr> m_runners;
    const int64_t m_canonical_stride;
    const int64_t m_batch_size;

    const int m_kmer_len;
    const bool m_is_reverse;

    // FIXME: remove on release
    const int64_t m_processed_chunks_max_size;
    utils::AsyncQueue<std::unique_ptr<ModBaseChunk>> m_processed_chunks;
    std::vector<std::unique_ptr<utils::AsyncQueue<std::unique_ptr<ModBaseChunk>>>> m_chunk_queues;
    std::mutex m_working_reads_mutex;

    std::vector<std::thread> m_runner_workers;
    std::atomic<int> m_num_active_runner_workers{0};

    std::vector<std::thread> m_output_workers;

    // Reads removed from input queue and being modbasecalled.
    std::unordered_set<std::shared_ptr<WorkingRead>> m_working_reads;

    // The offsets to the canonical bases in the modbase alphabet
    std::array<size_t, 4> m_base_prob_offsets;
    size_t m_num_states{4};
    std::shared_ptr<const ModBaseInfo> m_mod_base_info;

    // Performance monitoring stats.
    std::atomic<int64_t> m_num_batches_called{0};
    std::atomic<int64_t> m_num_partial_batches_called{0};
    std::atomic<int64_t> m_num_samples_processed{0};
    std::atomic<int64_t> m_num_samples_processed_incl_padding{0};

    // FIXME: Remove when done investigating
    const bool m_pad_tile{0};
    const bool m_pad_end_align{0};

    void validate_runners() const;

    bool populate_hits_seq(PerBaseIntVec& context_hits_seq,
                           const std::string& seq,
                           const modbase::RunnerPtr& runner) const;
    void populate_hits_sig(PerBaseIntVec& context_hits_sig,
                           const PerBaseIntVec& context_hits_seq,
                           const std::vector<uint64_t>& seq_to_sig_map) const;

    void populate_signal(at::Tensor& signal,
                         std::vector<uint64_t>& seq_to_sig_map,
                         const at::Tensor& raw_data,
                         const size_t signal_len,
                         const std::vector<int>& int_seq,
                         const modbase::RunnerPtr& runner) const;

    void populate_encoded_kmer(std::vector<int8_t>& encoded_kmer,
                               const size_t raw_samples,
                               const std::vector<int>& int_seq,
                               const std::vector<uint64_t>& seq_to_sig_map) const;

    void finalise_read(std::unique_ptr<dorado::SimplexRead>& read_ptr,
                       std::shared_ptr<WorkingRead>& working_read);

    std::vector<uint64_t> get_seq_to_sig_map(const std::vector<uint8_t>& moves,
                                             const size_t raw_samples,
                                             const size_t reserve) const;

    std::vector<ModBaseChunks> get_chunks(const modbase::RunnerPtr& runner,
                                          const std::shared_ptr<WorkingRead>& working_read) const;

    std::vector<std::pair<int64_t, int64_t>> get_chunk_contigs(
            const std::vector<ModBaseChunk>& chunks,
            const int64_t chunk_size) const;
};

}  // namespace dorado
