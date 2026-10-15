#pragma once

#include "ModelCaller.h"
#include "ModelRunnerBase.h"
#include "utils/stats.h"

#include <atomic>
#include <memory>
#include <string>

namespace dorado::basecall {

class ModelRunner final : public ModelRunnerBase {
public:
    explicit ModelRunner(std::shared_ptr<ModelCaller> caller, size_t batch_dims_idx);
    ~ModelRunner();

    void accept_chunk(int chunk_idx, const at::Tensor &chunk) final;
    std::vector<decode::DecodedChunk> call_chunks(int num_chunks) final;
    const config::BasecallModelConfig &config() const final;
    std::shared_ptr<ModBaseInfo> modbase_info() const final;
    size_t chunk_size() const final;
    size_t batch_size() const final;
    bool variable_chunk_sizes() const final;
    std::pair<int, int> batch_timeouts_ms() const final;
    bool is_low_latency() const final;
    void terminate() final;
    void restart() final;
    std::string get_name() const final;
    stats::NamedStats sample_stats() const final;

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

}  // namespace dorado::basecall
