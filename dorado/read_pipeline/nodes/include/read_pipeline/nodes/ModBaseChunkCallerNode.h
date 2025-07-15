#pragma once

#include "modbase/ModBaseRunner.h"
#include "read_pipeline/base/MessageSink.h"

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
    struct ModBaseData;
    struct WorkingRead;
    struct ModBaseChunk;

public:
    struct EncodingData {
        std::vector<uint64_t> seq_to_sig_map;
        std::vector<int> int_seq;
    };

    ModBaseChunkCallerNode(std::vector<modbase::RunnerPtr> model_runners,
                           size_t num_threads,
                           size_t canonical_stride,
                           size_t max_reads);
    ~ModBaseChunkCallerNode();

    std::string get_name() const override;
    stats::NamedStats sample_stats() const override;
    void terminate(const TerminateOptions&) override;
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

    static int64_t resolve_duplex_sequence_index(const int64_t resolved_score_index,
                                                 const int64_t target_start,
                                                 const int64_t sequence_size,
                                                 const bool is_template_direction);

    static std::vector<std::pair<int64_t, int64_t>> get_chunk_starts(
            const int64_t signal_len,
            const std::vector<int64_t>& hits_to_sig,
            const int64_t chunk_size,
            const int64_t context_samples_before,
            const int64_t context_samples_after,
            const bool end_align_last_chunk);

    static std::vector<bool> get_skip_positions(
            const std::vector<uint64_t>& seq_to_sig_map,
            const std::vector<std::pair<uint64_t, uint64_t>>& merged_chunks);

private:
    using ModBaseChunks = std::vector<std::unique_ptr<ModBaseChunkCallerNode::ModBaseChunk>>;
    using PerBaseIntVec = std::array<std::vector<int64_t>, 4>;

    void start_threads();
    void terminate_impl(utils::AsyncQueueTerminateFast fast);

    // Determine the modbase alphabet from all callers and calculate offset positions for the results
    void init_modbase_info();

    void input_thread_fn();

    void create_and_submit_chunks(modbase::RunnerPtr& runner,
                                  const size_t model_id,
                                  const int64_t previous_chunk_count,
                                  std::vector<std::unique_ptr<ModBaseChunk>>& batched_chunks) const;
    void chunk_caller_thread_fn(size_t worker_id, size_t model_id);

    void output_thread_fn();

    void simplex_mod_call(Message&& message);
    void duplex_mod_call(Message&& message);

    // Called by chunk_caller_thread_fn, calls the model and enqueues the results
    void call_batch(size_t worker_id, size_t model_id, ModBaseChunks& batched_chunks);

    std::vector<modbase::RunnerPtr> m_runners;
    const int64_t m_canonical_stride;
    const uint64_t m_sequence_stride_ratio;
    const int64_t m_batch_size;

    const int m_kmer_len;
    const bool m_is_rna_model;

    utils::AsyncQueue<std::unique_ptr<ModBaseChunk>> m_processed_chunks;
    std::vector<std::unique_ptr<utils::AsyncQueue<std::unique_ptr<ModBaseChunk>>>> m_chunk_queues;
    std::mutex m_working_reads_mutex;

    std::vector<std::thread> m_runner_workers;
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
    std::atomic<int64_t> m_num_samples_processed_incl_padding{0};

    std::atomic<size_t> m_num_chunks{0};
    std::atomic<int64_t> m_model_ms{0};
    std::atomic<int64_t> m_sequence_encode_ms{0};

    const bool m_pad_end_align{0};
    // Used to minimise the CPU work to encode CG contexts
    const bool m_minimal_encode{false};

    void validate_runners() const;

    void initialise_base_mod_probs(ReadCommon& read) const;

    std::optional<EncodingData> populate_modbase_data(ModBaseData& modbase_data,
                                                      const modbase::RunnerPtr& runner,
                                                      const std::string& seq,
                                                      const at::Tensor& signal,
                                                      const std::vector<uint8_t>& moves,
                                                      const std::string& read_id) const;

    bool populate_hits_seq(PerBaseIntVec& context_hits_seq,
                           const std::string& seq,
                           const modbase::RunnerPtr& runner) const;
    void populate_hits_sig(PerBaseIntVec& context_hits_sig,
                           const PerBaseIntVec& context_hits_seq,
                           const std::vector<uint64_t>& seq_to_sig_map) const;

    void populate_signal(at::Tensor& signal,
                         std::vector<uint64_t>& seq_to_sig_map,
                         const at::Tensor& raw_data,
                         const std::vector<int>& int_seq,
                         const modbase::RunnerPtr& runner) const;

    void populate_encoded_kmer(std::vector<int8_t>& encoded_kmer,
                               const size_t raw_samples,
                               const std::vector<int>& int_seq,
                               const std::vector<uint64_t>& seq_to_sig_map,
                               const std::vector<bool>& base_skips) const;

    std::vector<bool> get_minimal_encoding_skips(const modbase::RunnerPtr& runner,
                                                 const std::vector<ModBaseChunks>& chunks_by_caller,
                                                 const EncodingData& encoding_data) const;

    template <typename ReadType>
    void add_read_to_working_set(std::unique_ptr<ReadType> read_ptr,
                                 std::shared_ptr<WorkingRead> working_read);

    std::vector<uint64_t> get_seq_to_sig_map(const std::vector<uint8_t>& moves,
                                             const size_t raw_samples,
                                             const size_t reserve) const;

    std::vector<ModBaseChunks> get_chunks(const modbase::RunnerPtr& runner,
                                          const std::shared_ptr<WorkingRead>& working_read,
                                          const bool is_template) const;

    std::vector<std::pair<uint64_t, uint64_t>> merge_chunks(
            const std::vector<ModBaseChunks>& chunks_by_caller,
            const std::vector<uint64_t>& chunk_sizes) const;
};

}  // namespace dorado
