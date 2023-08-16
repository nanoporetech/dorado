#pragma once
#include "ReadPipeline.h"
#include "nn/CRFModelConfig.h"
#include "utils/stats.h"

#include <torch/torch.h>

#include <atomic>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace dorado {

class ScalerNode : public MessageSink {
public:
    ScalerNode(const SignalNormalisationParams& config,
               int num_worker_threads = 5,
               size_t max_reads = 1000);
    ~ScalerNode() { terminate_impl(); }
    std::string get_name() const override { return "ScalerNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions& flush_options) override { terminate_impl(); }
    void restart() override;

private:
    void start_threads();
    void terminate_impl();
    void worker_thread();  // Worker thread performs scaling and trimming asynchronously.
    std::vector<std::unique_ptr<std::thread>> m_worker_threads;
    std::atomic<int> m_num_worker_threads;

    SignalNormalisationParams m_scaling_params;

    std::pair<float, float> med_mad(const torch::Tensor& x);
    std::pair<float, float> normalisation(const torch::Tensor& x);
};

}  // namespace dorado
