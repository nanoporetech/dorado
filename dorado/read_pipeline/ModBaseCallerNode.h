#pragma once

#include "ReadPipeline.h"
#include "utils/AsyncQueue.h"
#include "utils/stats.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace dorado {

class ModBaseRunner;
struct RemoraChunk;
struct ModBaseParams;

class ModBaseCallerNode : public MessageSink {
public:
    ModBaseCallerNode(std::vector<std::unique_ptr<ModBaseRunner>> model_runners,
                      size_t remora_threads,
                      size_t block_stride,
                      size_t batch_size,
                      size_t max_reads = 1000);
    ~ModBaseCallerNode() { terminate_impl(); }
    std::string get_name() const override { return "ModBaseCallerNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate() override { terminate_impl(); }

    struct Info {
        std::string long_names;
        std::string alphabet;
    };

    // Expose long_names and alphabet computed by get_modbase_info
    static Info get_modbase_info(
            std::vector<std::reference_wrapper<ModBaseParams const>> const& base_mod_params) {
        return get_modbase_info_and_maybe_init(base_mod_params, nullptr);
    };

private:
    void terminate_impl();

    // Determine the modbase alphabet from parameters and calculate offset positions for the results
    // if node is not null it will populate its members
    [[maybe_unused]] static Info get_modbase_info_and_maybe_init(
            std::vector<std::reference_wrapper<ModBaseParams const>> const& base_mod_params,
            ModBaseCallerNode* node);

    // Determine the modbase alphabet from all callers and calculate offset positions for the results
    void init_modbase_info();

    // Worker threads, scales and chunks reads for runners and enqueues them
    void input_worker_thread();

    // Worker threads, performs the GPU calls to the modbase models
    void modbasecall_worker_thread(size_t worker_id, size_t caller_id);

    // Called by modbasecall_worker_thread, calls the model and enqueues the results
    void call_current_batch(size_t worker_id,
                            size_t caller_id,
                            std::vector<std::shared_ptr<RemoraChunk>>& batched_chunks);

    // Worker thread, processes chunk results back into the reads
    void output_worker_thread();

    size_t m_batch_size;
    size_t m_block_stride;

    std::vector<std::unique_ptr<ModBaseRunner>> m_runners;

    std::unique_ptr<std::thread> m_output_worker;
    std::vector<std::unique_ptr<std::thread>> m_runner_workers;
    std::vector<std::unique_ptr<std::thread>> m_input_worker;

    AsyncQueue<std::shared_ptr<RemoraChunk>> m_processed_chunks;
    std::vector<std::deque<std::shared_ptr<RemoraChunk>>> m_chunk_queues;

    std::mutex m_working_reads_mutex;
    // Reads removed from input queue and being modbasecalled.
    std::deque<std::shared_ptr<Read>> m_working_reads;

    std::mutex m_chunk_queues_mutex;
    std::condition_variable m_chunk_queues_cv;
    std::condition_variable m_chunks_added_cv;

    std::atomic<int> m_num_active_runner_workers{0};
    std::atomic<int> m_num_active_input_worker{0};

    std::atomic<bool> m_terminate_runners{false};
    std::atomic<bool> m_terminate_output{false};

    std::shared_ptr<const utils::BaseModInfo> m_base_mod_info;
    // The offsets to the canonical bases in the modbase alphabet
    std::array<size_t, 4> m_base_prob_offsets;
    size_t m_num_states{4};

    // Performance monitoring stats.
    std::atomic<int64_t> m_num_batches_called = 0;
    std::atomic<int64_t> m_num_partial_batches_called = 0;
    std::atomic<int64_t> m_num_input_chunks_sleeps = 0;
    std::atomic<int64_t> m_call_chunks_ms = 0;
    std::atomic<int64_t> m_num_context_hits = 0;
    std::atomic<int64_t> m_num_mod_base_reads_pushed = 0;
    std::atomic<int64_t> m_num_non_mod_base_reads_pushed = 0;
    std::atomic<int64_t> m_chunk_generation_ms = 0;
};

}  // namespace dorado
