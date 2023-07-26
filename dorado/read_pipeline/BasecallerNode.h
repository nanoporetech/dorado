#pragma once

#include "../nn/ModelRunner.h"
#include "ReadPipeline.h"
#include "utils/AsyncQueue.h"
#include "utils/stats.h"

#include <atomic>
#include <cstdint>
#include <unordered_set>

namespace dorado {

class BasecallerNode : public MessageSink {
public:
    // Chunk size and overlap are in raw samples
    BasecallerNode(std::vector<Runner> model_runners,
                   size_t overlap,
                   int batch_timeout_ms,
                   std::string model_name = "",
                   size_t max_reads = 1000,
                   const std::string& node_name = "BasecallerNode",
                   bool in_duplex_pipeline = false,
                   uint32_t read_mean_qscore_start_pos = 0);
    ~BasecallerNode() { terminate_impl(); }
    std::string get_name() const override { return m_node_name; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions& flush_options) override { terminate_impl(); }
    void restart() override;

private:
    void start_threads();
    void terminate_impl();
    // Consume reads from input queue
    void input_worker_thread();
    // Basecall reads
    void basecall_worker_thread(int worker_id);
    // Basecall batch of chunks
    void basecall_current_batch(int worker_id);
    // Construct complete reads
    void working_reads_manager();

    // Vector of model runners (each with their own GPU access etc)
    std::vector<Runner> m_model_runners;
    // Chunk length
    size_t m_chunk_size;
    // Minimum overlap between two adjacent chunks in a read. Overlap is used to reduce edge effects and improve accuracy.
    size_t m_overlap;
    // Stride of the model in the runners
    size_t m_model_stride;
    // Time in milliseconds before partial batches are called.
    int m_batch_timeout_ms;
    // model_name
    std::string m_model_name;
    // max reads
    size_t m_max_reads;
    // is this node part of a duplex pipeline?
    bool m_in_duplex_pipeline;
    // Mean Q-score start position from model properties.
    uint32_t m_mean_qscore_start_pos;

    // Model runners which have not terminated.
    std::atomic<int> m_num_active_model_runners{0};

    // Time when Basecaller Node is initialised. Used for benchmarking and debugging
    std::chrono::time_point<std::chrono::system_clock> initialization_time;
    // Time when Basecaller Node terminates. Used for benchmarking and debugging
    std::chrono::time_point<std::chrono::system_clock> termination_time;
    // Async queue to keep track of basecalling chunks.
    utils::AsyncQueue<std::shared_ptr<Chunk>> m_chunks_in;

    std::mutex m_working_reads_mutex;
    // Reads removed from input queue and being basecalled.
    std::unordered_set<std::shared_ptr<Read>> m_working_reads;

    // If we go multi-threaded, there will be one of these batches per thread
    std::vector<std::deque<std::shared_ptr<Chunk>>> m_batched_chunks;

    utils::AsyncQueue<std::shared_ptr<Chunk>> m_processed_chunks;

    // Class members are initialised in declaration order regardless of initialiser list order.
    // Class data members whose construction launches threads must therefore have their
    // declarations follow those of the state on which they rely, e.g. mutexes, if their
    // initialisation is via initialiser lists.
    // Chunks up incoming reads and sticks them in the pending list.
    std::unique_ptr<std::thread> m_input_worker;
    // Basecalls chunks from the queue and puts read on the sink.
    std::vector<std::thread> m_basecall_workers;
    // Stitches working reads into complete reads.
    std::vector<std::thread> m_working_reads_managers;

    // Performance monitoring stats.
    std::string m_node_name;
    std::atomic<int64_t> m_num_batches_called = 0;
    std::atomic<int64_t> m_num_partial_batches_called = 0;
    std::atomic<int64_t> m_call_chunks_ms = 0;
    std::atomic<int64_t> m_called_reads_pushed = 0;
    std::atomic<int64_t> m_working_reads_size = 0;
    std::atomic<int64_t> m_num_bases_processed = 0;
    std::atomic<int64_t> m_num_samples_processed = 0;
};

}  // namespace dorado
