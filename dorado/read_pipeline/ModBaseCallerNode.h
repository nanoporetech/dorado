#pragma once
#include "ReadPipeline.h"

#include <atomic>
#include <deque>
#include <memory>
#include <vector>

class RemoraChunk;
class RemoraCaller;

class ModBaseCallerNode : public ReadSink {
public:
    ModBaseCallerNode(ReadSink& sink,
                      std::vector<std::shared_ptr<RemoraCaller>> model_callers,
                      size_t remora_threads,
                      size_t block_stride,
                      size_t batch_size,
                      size_t max_reads = 1000);
    ~ModBaseCallerNode();

private:
    void init_modbase_info();

    void input_worker_thread();   // Worker thread distributes reads to the runners.
    void output_worker_thread();  // Worker thread processes results into the reads.

    void runner_worker_thread(int runner_id);
    void caller_worker_thread(int caller_id);

    void call_current_batch(int caller_id);

    ReadSink& m_sink;
    size_t m_batch_size;
    size_t m_block_stride;

    std::vector<std::shared_ptr<RemoraCaller>> m_callers;

    std::unique_ptr<std::thread> m_output_worker;
    std::vector<std::unique_ptr<std::thread>> m_caller_workers;
    std::vector<std::unique_ptr<std::thread>> m_runner_workers;

    std::deque<std::shared_ptr<RemoraChunk>> m_processed_chunks;
    std::vector<std::deque<std::shared_ptr<RemoraChunk>>> m_batched_chunks;
    std::vector<std::deque<std::shared_ptr<RemoraChunk>>> m_chunk_queues;

    std::mutex m_working_reads_mutex;
    // Reads removed from input queue and being modbasecalled.
    std::deque<std::shared_ptr<Read>> m_working_reads;

    std::mutex m_chunk_queues_mutex;
    std::condition_variable m_chunk_queues_cv;

    std::mutex m_processed_chunks_mutex;
    std::condition_variable m_processed_chunks_cv;

    std::atomic<int> m_num_active_model_callers{0};
    std::atomic<int> m_num_active_model_runners{0};

    bool m_terminate_callers{false};
    bool m_terminate_output{false};

    using BaseModInfo = ::utils::BaseModInfo;
    std::shared_ptr<const BaseModInfo> m_base_mod_info;
    // The offsets to the canonical bases in the modbase alphabet
    std::array<size_t, 4> m_base_prob_offsets;
    size_t m_num_states{4};
};