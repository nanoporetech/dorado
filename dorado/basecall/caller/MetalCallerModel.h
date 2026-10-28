#pragma once

#include "CallerModel.h"
#include "MetalCallerTask.h"
#include "TensorRunnerState.h"
#include "config/BasecallModelConfig.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace dorado::basecall {

class MetalModel;

// Adapts a MetalModel to a CallerModel. This owns the Metal-specific parts of running a batch:
// command buffer submission with retries, the decode completion event, and autorelease pools.
class MetalCallerModel final
        : public TypedCallerModel<MetalCallerModel, TensorRunnerState, MetalCallerTask> {
public:
    MetalCallerModel(const config::BasecallModelConfig &model_config,
                     std::unique_ptr<MetalModel> model);
    ~MetalCallerModel() override;

    const config::BasecallModelConfig &config() const override { return m_config; }
    std::unique_ptr<RunnerState> create_runner_state(size_t batch_dims_idx) override;
    std::shared_ptr<ModBaseInfo> modbase_info() const override;
    std::string name() const override { return "MetalCaller"; }

    SchedulingPolicy scheduling_policy() const override;
    std::string worker_thread_name() const override { return "metal_worker"; }
    std::string decode_thread_name() const override { return "metal_decode"; }
    std::unique_ptr<ThreadScope> worker_thread_scope() override;

private:
    friend class TypedCallerModel<MetalCallerModel, TensorRunnerState, MetalCallerTask>;
    std::unique_ptr<MetalCallerTask> prepare_batch(TensorRunnerState &state, int num_chunks);
    void submit_batch(MetalCallerTask &task);
    size_t decode_units_batch(const MetalCallerTask &task) const;
    void decode_batch(MetalCallerTask &task, size_t unit, std::vector<decode::DecodedChunk> &out);
    void release_batch(MetalCallerTask &task, std::vector<decode::DecodedChunk> &out);

    const config::BasecallModelConfig m_config;
    std::unique_ptr<MetalModel> m_model;
};

}  // namespace dorado::basecall
