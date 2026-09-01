#pragma once

#include "config/BasecallModelConfig.h"
#include "read_pipeline/base/MessageSink.h"

#include <string>

namespace dorado {

class ScalerNode : public MessageSink {
public:
    ScalerNode(const config::SignalNormalisationParams& config,
               models::SampleType model_type,
               int num_worker_threads,
               size_t max_reads);
    ~ScalerNode();

    std::string get_name() const override;
    void terminate(const TerminateOptions&) override;
    void restart() override;

private:
    ScalerNode(const ScalerNode&) = delete;
    ScalerNode& operator=(const ScalerNode&) = delete;
    ScalerNode(ScalerNode&&) = delete;
    ScalerNode& operator=(ScalerNode&&) = delete;

private:
    void input_thread_fn();

    const config::SignalNormalisationParams m_scaling_params;
    const models::SampleType m_model_type;
};

}  // namespace dorado
