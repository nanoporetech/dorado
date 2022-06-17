#pragma once
#include "../nn/ModelRunner.h"
#include "ReadPipeline.h"

class BasecallerNode : public ReadSink {
public:
    // Chunk size and overlap are in raw samples
    BasecallerNode(ReadSink &sink,
                   std::vector<Runner> &model_runners,
                   size_t batch_size,
                   size_t chunk_size,
                   size_t overlap,
                   size_t max_reads = 1000);
    ~BasecallerNode();

private:
    // Consume reads from input queue
    void input_worker_thread();
    // Basecall reads
    void basecall_worker_thread(int worker_id);
    // Basecall batch of chunks
    void basecall_current_batch(int worker_id);

    ReadSink &m_sink;
    std::unique_ptr<std::thread>
            m_input_worker;  // Chunks up incoming reads and sticks them in the pending list
    std::vector<std::unique_ptr<std::thread>>
            m_basecall_workers;  // Basecalls chunks from the queue and puts read on the sink.
    std::string m_model_path;
    // Vector of model runners (each with their own GPU access etc)
    std::vector<Runner> m_model_runners;
    // Number of chunks in a batch
    size_t m_batch_size;
    // Chunk length
    size_t m_chunk_size;
    // Minimum overlap between two adjacent chunks in a read. Overlap is used to reduce edge effects and improve accuracy.
    size_t m_overlap;

    // Model runners which have not terminated.
    std::atomic<int> m_num_active_model_runners = 0;

    // Time when Basecaller Node is initialised. Used for benchmarking and debugging
    std::chrono::time_point<std::chrono::system_clock> initialization_time;
    // Time when Basecaller Node terminates. Used for benchmarking and debugging
    std::chrono::time_point<std::chrono::system_clock> termination_time;
    // Global chunk input list
    std::mutex m_chunks_in_mutex;
    // Gets filled with chunks from the input reads
    std::deque<std::shared_ptr<Chunk>> m_chunks_in;

    std::mutex m_working_reads_mutex;
    // Reads removed from input queue and being basecalled.
    std::deque<std::shared_ptr<Read>> m_working_reads;

    // If we go multi-threaded, there will be one of these batches per thread
    std::vector<std::deque<std::shared_ptr<Chunk>>> m_batched_chunks;

    bool m_terminate_basecaller;
};
