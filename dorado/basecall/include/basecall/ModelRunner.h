#pragma once

#include "ModelCaller.h"
#include "utils/stats.h"

#include <atomic>
#include <memory>
#include <string>

namespace dorado::basecall {

class ModelRunner final {
public:
    explicit ModelRunner(std::shared_ptr<ModelCaller> caller, size_t batch_dims_idx);
    ~ModelRunner();

    void accept_chunk(int chunk_idx, const at::Tensor &chunk);
    std::vector<decode::DecodedChunk> call_chunks(int num_chunks);
    const config::BasecallModelConfig &config() const;
    std::shared_ptr<ModBaseInfo> modbase_info() const;
    size_t chunk_size() const;
    size_t batch_size() const;
    bool variable_chunk_sizes() const;
    std::pair<int, int> batch_timeouts_ms() const;
    bool is_low_latency() const;
    void terminate();
    void restart();
    std::string get_name() const;
    stats::NamedStats sample_stats() const;

private:
    ModelRunner(const ModelRunner &) = delete;
    ModelRunner &operator=(const ModelRunner &) = delete;
    ModelRunner(ModelRunner &&) = delete;
    ModelRunner &operator=(ModelRunner &&) = delete;

private:
    std::shared_ptr<ModelCaller> m_caller;
    std::unique_ptr<ModelCaller::RunnerContext> m_runner_context;

    // Performance monitoring stats.
    std::atomic<int64_t> m_num_batches_called = 0;
};

using RunnerPtr = std::unique_ptr<ModelRunner>;

size_t auto_calculate_num_runners(const config::BasecallModelConfig &model_config,
                                  float memory_fraction);

}  // namespace dorado::basecall
