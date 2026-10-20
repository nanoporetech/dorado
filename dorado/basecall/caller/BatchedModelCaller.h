#pragma once

#include "BatchExecutor.h"
#include "CallerModel.h"
#include "basecall/ModelCaller.h"

#include <memory>
#include <string>

namespace dorado::basecall {

// A ModelCaller which runs batches through a CallerModel using a BatchExecutor.
class BatchedModelCaller final : public ModelCaller {
public:
    explicit BatchedModelCaller(std::unique_ptr<CallerModel> model);
    ~BatchedModelCaller() override;

    std::unique_ptr<RunnerContext> create_runner_context(size_t batch_dims_idx) override;
    std::vector<decode::DecodedChunk> call_chunks(RunnerState& state, int num_chunks);

    size_t num_batch_dims() const override { return m_model->num_batch_dims(); }
    const config::BasecallModelConfig& config() const override { return m_model->config(); }
    std::shared_ptr<ModBaseInfo> modbase_info() const override { return m_model->modbase_info(); }
    std::pair<int, int> batch_timeouts_ms() const override { return m_model->batch_timeouts_ms(); }
    bool is_low_latency() const override { return m_model->is_low_latency(); }
    void terminate() override { m_executor.terminate(); }
    void restart() override { m_executor.restart(); }
    std::string get_name() const override { return m_model->name(); }
    stats::NamedStats sample_stats() const override { return m_model->sample_stats(); }

private:
    std::unique_ptr<CallerModel> m_model;
    BatchExecutor m_executor;
};

}  // namespace dorado::basecall
