#pragma once

#include "DecodedChunk.h"
#include "utils/stats.h"
#include "utils/types.h"

#include <cstddef>
#include <memory>
#include <span>
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

class BatchExecutor;
class CallerModel;
class RunnerState;

// Runs batches of chunks through a model. All backend-specific behaviour is provided by the
// CallerModel; queueing, threading and decoding are handled by a BatchExecutor.
class ModelCaller final {
public:
    // Accepts chunks for one runner. It does not own or refer to the ModelCaller which created it,
    // so must be passed that caller to call its chunks as a batch.
    class RunnerContext final {
    public:
        explicit RunnerContext(std::unique_ptr<RunnerState> state);
        ~RunnerContext();

        void accept_chunk(int chunk_idx, const at::Tensor& chunk);
        std::vector<decode::DecodedChunk> call_chunks(ModelCaller& caller, int num_chunks);
        size_t chunk_size() const;
        size_t batch_size() const;
        bool variable_chunk_sizes() const;

    private:
        std::unique_ptr<RunnerState> m_state;
    };

    explicit ModelCaller(std::unique_ptr<CallerModel> model);
    ~ModelCaller();

    ModelCaller(const ModelCaller&) = delete;
    ModelCaller& operator=(const ModelCaller&) = delete;

    std::unique_ptr<RunnerContext> create_runner_context(size_t batch_dims_idx);
    std::vector<decode::DecodedChunk> call_chunks(RunnerState& state, int num_chunks);

    size_t num_batch_dims() const;
    const config::BasecallModelConfig& config() const;
    std::shared_ptr<ModBaseInfo> modbase_info() const;
    std::pair<int, int> batch_timeouts_ms() const;
    bool is_low_latency() const;
    void terminate();
    void restart();
    std::string get_name() const;
    stats::NamedStats sample_stats() const;

private:
    std::unique_ptr<CallerModel> m_model;
    std::unique_ptr<BatchExecutor> m_executor;
};

}  // namespace dorado::basecall
