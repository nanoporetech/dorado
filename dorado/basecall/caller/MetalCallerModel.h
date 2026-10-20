#pragma once

#include "CallerModel.h"
#include "config/BasecallModelConfig.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace dorado::basecall {

class MetalModel;

// Adapts a MetalModel to a CallerModel. This owns the Metal-specific parts of running a batch:
// command buffer submission with retries, the decode completion event, and autorelease pools.
class MetalCallerModel final : public CallerModel {
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

    std::unique_ptr<InFlightBatch> prepare(RunnerState &state, int num_chunks) override;
    void submit(InFlightBatch &batch) override;
    size_t decode_units(const InFlightBatch &batch) const override;
    void decode(InFlightBatch &batch, size_t unit, std::vector<decode::DecodedChunk> &out) override;
    void release(InFlightBatch &batch, std::vector<decode::DecodedChunk> &out) override;

private:
    const config::BasecallModelConfig m_config;
    std::unique_ptr<MetalModel> m_model;

    // Incrementing ID used to prevent the linear layer of run i+1 overwriting the scores of
    // run i before the CPU has finished decoding all run i's chunks. Start at 1, since at event
    // creation ID 0 is deemed to have been signalled. Only accessed from the worker thread.
    uint64_t m_next_decode_complete_event_id{1};
};

}  // namespace dorado::basecall
