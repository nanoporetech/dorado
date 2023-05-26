#pragma once

#include "../nn/ModelRunner.h"
#include "ReadPipeline.h"
#include "utils/stats.h"

#include <atomic>
#include <cstdint>

namespace dorado {

class BasecallerNode : public MessageSink {
public:
    // Chunk size and overlap are in raw samples
    BasecallerNode(MessageSink &sink,
                   std::vector<Runner> model_runners,
                   size_t overlap,
                   int batch_timeout_ms,
                   std::string model_name = "",
                   size_t max_reads = 1000);
    ~BasecallerNode();
    std::string get_name() const override { return "BasecallerNode"; }
    stats::NamedStats sample_stats() const override;

private:
    // Consume reads from input queue
    void input_worker_thread();
    // Basecall reads
    void basecall_worker_thread(int worker_id);
    // Basecall batch of chunks
    void basecall_current_batch(int worker_id);
    // Construct complete reads
    void working_reads_manager();

    MessageSink &m_sink;
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

    // Model runners which have not terminated.
    std::atomic<int> m_num_active_model_runners{0};

    std::atomic<bool> m_terminate_basecaller{false};
    std::atomic<bool> m_terminate_manager{false};

    // Time when Basecaller Node is initialised. Used for benchmarking and debugging
    std::chrono::time_point<std::chrono::system_clock> initialization_time;
    // Time when Basecaller Node terminates. Used for benchmarking and debugging
    std::chrono::time_point<std::chrono::system_clock> termination_time;
    // Signalled when there is space in m_chunks_in
    std::condition_variable m_chunks_in_has_space_cv;
    // Global chunk input list
    std::mutex m_chunks_in_mutex;
    // Gets filled with chunks from the input reads
    std::deque<std::shared_ptr<Chunk>> m_chunks_in;

    std::mutex m_working_reads_mutex;
    // Reads removed from input queue and being basecalled.
    std::deque<std::shared_ptr<Read>> m_working_reads;

    // If we go multi-threaded, there will be one of these batches per thread
    std::vector<std::deque<std::shared_ptr<Chunk>>> m_batched_chunks;

    // Class members are initialised in declaration order regardless of initialiser list order.
    // Class data members whose construction launches threads must therefore have their
    // declarations follow those of the state on which they rely, e.g. mutexes, if their
    // initialisation is via initialiser lists.
    std::unique_ptr<std::thread>
            m_input_worker;  // Chunks up incoming reads and sticks them in the pending list.
    std::vector<std::thread>
            m_basecall_workers;  // Basecalls chunks from the queue and puts read on the sink.
    std::unique_ptr<std::thread>
            m_working_reads_manager;  // Stitches working reads into complete reads.

    // Performance monitoring stats.
    std::atomic<int64_t> m_num_batches_called = 0;
    std::atomic<int64_t> m_num_partial_batches_called = 0;
    std::atomic<int64_t> m_num_input_chunks_sleeps = 0;
    std::atomic<int64_t> m_call_chunks_ms = 0;
    std::atomic<int64_t> m_called_reads_pushed = 0;
};

}  // namespace dorado
