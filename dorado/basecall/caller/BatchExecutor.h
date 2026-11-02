#pragma once

#include "CallerModel.h"
#include "basecall/DecodedChunk.h"

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace dorado::basecall {

class BatchExecutor;

// A batch waiting for, or going through, the executor pipeline.
struct BatchTask {
    BatchTask(const BatchExecutor* owner_, std::unique_ptr<InFlightBatch> work_, int num_chunks_)
            : owner(owner_),
              work(std::move(work_)),
              num_chunks(num_chunks_),
              out(static_cast<size_t>(num_chunks_)) {}

    const BatchExecutor* owner;
    std::unique_ptr<InFlightBatch> work;
    const int num_chunks;
    std::vector<decode::DecodedChunk> out;

    std::mutex mut;
    std::condition_variable cv;
    // The thread waiting in execute() can proceed: submit() has returned if the requesting thread
    // decodes, otherwise all decode units have been decoded and the batch released.
    bool ready{false};
    size_t decode_units{0};
    size_t decode_units_finished{0};
    // Guarded by the decode queue lock.
    size_t decode_units_started{0};
};

// A FIFO of batches awaiting submission. Executors for models which report the same
// SchedulingPolicy::shared_queue_key share a queue, so that they submit one batch at a time, in
// the order that the batches became ready.
class DeviceQueue {
public:
    // Returns the queue for `key`, or a new private queue if `key` is empty.
    static std::shared_ptr<DeviceQueue> get(const std::string& key);

    std::mutex mut;
    std::condition_variable cv;
    std::deque<std::shared_ptr<BatchTask>> queue;
};

// Runs batches through a CallerModel on a dedicated worker thread, with optional decode threads.
class BatchExecutor final {
public:
    explicit BatchExecutor(CallerModel& model);
    ~BatchExecutor();

    BatchExecutor(const BatchExecutor&) = delete;
    BatchExecutor& operator=(const BatchExecutor&) = delete;

    // Queues a prepared batch, and blocks until it has been decoded.
    std::vector<decode::DecodedChunk> execute(std::unique_ptr<InFlightBatch> work, int num_chunks);

    void terminate();
    void restart();

private:
    void start_threads();
    void worker_thread_fn();
    void decode_thread_fn();
    void decode_inline(BatchTask& task);
    static void mark_ready(BatchTask& task);

    CallerModel& m_model;
    const SchedulingPolicy m_policy;
    std::shared_ptr<DeviceQueue> m_input_queue;
    // Guarded by m_input_queue->mut.
    bool m_terminate{false};
    std::thread m_worker_thread;

    std::deque<std::shared_ptr<BatchTask>> m_decode_queue;
    std::mutex m_decode_lock;
    std::condition_variable m_decode_cv;
    bool m_terminate_decode{false};
    std::vector<std::thread> m_decode_threads;
};

}  // namespace dorado::basecall
