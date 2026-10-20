#pragma once

#include "CudaModel.h"
#include "basecall/DecodedChunk.h"
#include "basecall/ModelCaller.h"
#include "basecall/ModelCallerFactory.h"
#include "config/BasecallModelConfig.h"
#include "nn/AuxiliaryData.h"
#include "utils/stats.h"

#include <torch/nn.h>

#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace dorado::basecall {

class CudaCaller final : public ModelCaller {
public:
    CudaCaller(const BasecallerCreationParams &params);

    ~CudaCaller() override;
    std::unique_ptr<RunnerContext> create_runner_context(size_t batch_dims_idx) override;
    std::vector<decode::DecodedChunk> call_chunks(CudaRunnerState &context, int num_chunks);
    std::vector<decode::DecodedChunk> call_chunks(at::Tensor &input,
                                                  at::Tensor &output,
                                                  int num_chunks,
                                                  nn::AuxiliaryData *aux /* = nullptr */);

    void terminate() override;
    void restart() override;

    size_t num_batch_dims() const override { return m_model->num_batch_dims(); };
    const config::BasecallModelConfig &config() const override { return m_model->config(); }
    std::shared_ptr<ModBaseInfo> modbase_info() const override { return m_model->modbase_info(); }
    bool is_low_latency() const override { return m_model->is_low_latency(); }
    std::string get_name() const override {
        return std::string("CudaCaller_") + m_model->device_name();
    }
    stats::NamedStats sample_stats() const override { return m_model->sample_stats(); }
    std::pair<int, int> batch_timeouts_ms() const override;

private:
    struct GPUTaskQueue;
    GPUTaskQueue &get_task_queue();

    void start_threads();
    void cuda_thread_fn();

    std::unique_ptr<CudaModel> m_model;
    bool m_terminate{false};
    std::thread m_cuda_thread;
};

}  // namespace dorado::basecall
