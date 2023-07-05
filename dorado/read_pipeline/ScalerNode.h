#pragma once
#include "ReadPipeline.h"
#include "nn/CRFModel.h"
#include "utils/stats.h"

#include <atomic>
#include <string>
#include <thread>
#include <vector>

namespace dorado {

class ScalerNode : public MessageSink {
public:
    ScalerNode(MessageSink& sink,
               const SignalNormalisationParams& config,
               int num_worker_threads = 5,
               size_t max_reads = 1000);
    ~ScalerNode();
    std::string get_name() const override { return "ScalerNode"; }
    stats::NamedStats sample_stats() const override;

private:
    void worker_thread();  // Worker thread performs scaling and trimming asynchronously.
    MessageSink&
            m_sink;  // MessageSink to consume scaled reads. Typically this will be a Basecaller Node.
    std::vector<std::unique_ptr<std::thread>> worker_threads;
    std::atomic<int> m_num_worker_threads;

    SignalNormalisationParams m_scaling_params;

    std::pair<float, float> med_mad(torch::Tensor& x, float factor);
    std::pair<float, float> normalisation(torch::Tensor& x);
};

}  // namespace dorado
