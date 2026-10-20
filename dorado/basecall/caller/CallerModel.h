#pragma once

#include "basecall/DecodedChunk.h"
#include "utils/stats.h"
#include "utils/types.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace at {
class Tensor;
}

namespace dorado::config {
struct BasecallModelConfig;
}

namespace dorado::basecall {

// Per-runner input buffers and chunk bookkeeping. Created by, and only ever passed back to, the
// CallerModel which created it.
class RunnerState {
public:
    virtual ~RunnerState() = default;
    virtual void accept_chunk(int chunk_idx, const at::Tensor& chunk) = 0;
    virtual size_t chunk_size() const = 0;
    virtual size_t batch_size() const = 0;
    virtual bool variable_chunk_sizes() const { return false; }

protected:
    RunnerState() = default;
};

// Opaque state for a batch as it moves through the BatchExecutor pipeline. Each CallerModel
// defines its own derived type.
class InFlightBatch {
public:
    virtual ~InFlightBatch() = default;

protected:
    InFlightBatch() = default;
};

// RAII object kept alive for the lifetime of an executor thread, for per-thread setup such as
// stream guards or autorelease pools.
class ThreadScope {
public:
    virtual ~ThreadScope() = default;

protected:
    ThreadScope() = default;
};

// When a batch leaves the queue it was waiting in.
enum class ReleasePoint {
    on_submit_start,   // Removed from the queue before submit() is called.
    on_submit_return,  // Remains at the front of the queue until submit() has returned.
};

struct SchedulingPolicy {
    // Callers whose models report the same non-empty key share one FIFO queue, so that their
    // batches are submitted one at a time in the order they became ready. An empty key gives the
    // caller a private queue.
    std::string shared_queue_key;
    ReleasePoint release_point{ReleasePoint::on_submit_start};
    // Number of threads decoding batch units. If zero, the thread which requested the batch
    // decodes it.
    size_t decode_threads{0};
};

// The backend-specific part of a model caller. A BatchExecutor drives a batch through:
//   prepare()  - on the thread requesting the batch.
//   submit()   - on the executor's worker thread, serialised according to the SchedulingPolicy.
//   decode()   - once per decode unit, on the requesting thread or the decode pool.
//   release()  - once all decode units have completed.
class CallerModel {
public:
    virtual ~CallerModel() = default;

    virtual const config::BasecallModelConfig& config() const = 0;
    virtual std::shared_ptr<ModBaseInfo> modbase_info() const { return nullptr; }
    virtual size_t num_batch_dims() const { return 1; }
    virtual std::unique_ptr<RunnerState> create_runner_state(size_t batch_dims_idx) = 0;
    virtual std::pair<int, int> batch_timeouts_ms() const;
    virtual bool is_low_latency() const { return false; }
    virtual std::string name() const = 0;
    virtual stats::NamedStats sample_stats() const { return {}; }

    virtual SchedulingPolicy scheduling_policy() const { return {}; }
    virtual std::string worker_thread_name() const { return "model_caller"; }
    virtual std::string decode_thread_name() const { return "model_decode"; }
    virtual std::unique_ptr<ThreadScope> worker_thread_scope() { return nullptr; }

    virtual std::unique_ptr<InFlightBatch> prepare(RunnerState& state, int num_chunks) = 0;
    virtual void submit(InFlightBatch& batch) = 0;
    virtual size_t decode_units(const InFlightBatch& batch) const = 0;
    // `out` has one entry per chunk in the batch. Each unit fills in the entries it is
    // responsible for.
    virtual void decode(InFlightBatch& batch,
                        size_t unit,
                        std::vector<decode::DecodedChunk>& out) = 0;
    virtual void release(InFlightBatch& batch) { (void)batch; }

protected:
    CallerModel() = default;
};

}  // namespace dorado::basecall
