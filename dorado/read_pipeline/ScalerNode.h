#pragma once
#include "ReadPipeline.h"
#include "basecall/CRFModelConfig.h"
#include "utils/stats.h"

#include <ATen/core/TensorBody.h>

#include <atomic>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace dorado {

class ScalerNode : public MessageSink {
public:
    ScalerNode(const basecall::SignalNormalisationParams& config,
               basecall::SampleType model_type,
               bool trim_adapter,
               int num_worker_threads,
               size_t max_reads);
    ~ScalerNode() { terminate_impl(); }
    std::string get_name() const override { return "ScalerNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions&) override { terminate_impl(); }
    void restart() override;

private:
    void start_threads();
    void terminate_impl();
    void worker_thread();  // Worker thread performs scaling and trimming asynchronously.
    std::vector<std::unique_ptr<std::thread>> m_worker_threads;
    std::atomic<int> m_num_worker_threads;

    basecall::SignalNormalisationParams m_scaling_params;
    const basecall::SampleType m_model_type;
    const bool m_trim_adapter;

    std::pair<float, float> med_mad(const at::Tensor& x);
    std::pair<float, float> normalisation(const at::Tensor& x);
};

}  // namespace dorado
