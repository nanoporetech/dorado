#pragma once

#include "MessageSink.h"
#include "utils/AsyncQueue.h"
#include "utils/stats.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace dorado {

namespace basecall {
class ModelRunnerBase;
using RunnerPtr = std::unique_ptr<ModelRunnerBase>;
}  // namespace basecall

class BasecallerNode : public MessageSink {
    struct BasecallingRead;
    struct BasecallingChunk;

public:
    // Chunk size and overlap are in raw samples
    BasecallerNode(std::vector<basecall::RunnerPtr> model_runners,
                   size_t overlap,
                   std::string model_name,
                   size_t max_reads,
                   std::string node_name,
                   uint32_t read_mean_qscore_start_pos);
    ~BasecallerNode();
    std::string get_name() const override { return m_node_name; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions &) override { terminate_impl(); }
    void restart() override;

private:
    void start_threads();
    void terminate_impl();
    // Consume reads from input queue, chunks them up, and sticks them in the pending list.
    void input_thread_fn();
    // Basecall reads
    void basecall_worker_thread(int worker_id);
    // Basecall batch of chunks
    void basecall_current_batch(int worker_id);
    // Construct complete reads
    void working_reads_manager();

    size_t get_chunk_queue_idx(size_t read_raw_size);

    // Override for batch timeout to use for low-latency pipelines. Zero means use the normal timeout.
    int m_low_latency_batch_timeout_ms;
    // Vector of model runners (each with their own GPU access etc)
    std::vector<basecall::RunnerPtr> m_model_runners;
    // Minimum overlap between two adjacent chunks in a read. Overlap is used to reduce edge effects and improve accuracy.
    size_t m_overlap;
    // Stride of the model in the runners
    size_t m_model_stride;
    // Whether the model is for rna
    bool m_is_rna_model;
    // model_name
    std::string m_model_name;
    // Mean Q-score start position from model properties.
    uint32_t m_mean_qscore_start_pos;

    // Model runners which have not terminated.
    std::atomic<int> m_num_active_model_runners{0};

    // Async queues to keep track of basecalling chunks. Each queue is for a different chunk size.
    // Basecall worker threads map to queue: `m_chunk_in_queues[worker_id % m_chunk_sizes.size()]`
    std::vector<size_t> m_chunk_sizes;
    std::vector<std::unique_ptr<utils::AsyncQueue<std::unique_ptr<BasecallingChunk>>>>
            m_chunk_in_queues;

    std::mutex m_working_reads_mutex;
    // Reads removed from input queue and being basecalled.
    std::unordered_set<std::shared_ptr<BasecallingRead>> m_working_reads;

    // If we go multi-threaded, there will be one of these batches per thread
    std::vector<std::vector<std::unique_ptr<BasecallingChunk>>> m_batched_chunks;

    utils::AsyncQueue<std::unique_ptr<BasecallingChunk>> m_processed_chunks;

    // Class members are initialised in declaration order regardless of initialiser list order.
    // Class data members whose construction launches threads must therefore have their
    // declarations follow those of the state on which they rely, e.g. mutexes, if their
    // initialisation is via initialiser lists.
    // Basecalls chunks from the queue and puts read on the sink.
    std::vector<std::thread> m_basecall_workers;
    // Stitches working reads into complete reads.
    std::vector<std::thread> m_working_reads_managers;

    // Performance monitoring stats.
    const std::string m_node_name;
    std::atomic<int64_t> m_num_batches_called = 0;
    std::atomic<int64_t> m_num_partial_batches_called = 0;
    std::atomic<int64_t> m_call_chunks_ms = 0;
    std::atomic<int64_t> m_called_reads_pushed = 0;
    std::atomic<int64_t> m_working_reads_size = 0;
    std::atomic<int64_t> m_num_bases_processed = 0;
    std::atomic<int64_t> m_num_samples_processed = 0;
    std::atomic<int64_t> m_num_samples_incl_padding = 0;
    std::atomic<int64_t> m_working_reads_signal_bytes = 0;
};

}  // namespace dorado
