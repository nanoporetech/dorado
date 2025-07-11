#pragma once

#include "config/BasecallModelConfig.h"
#include "read_pipeline/base/MessageSink.h"

#include <atomic>
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
    void input_thread_fn();

    const config::SignalNormalisationParams m_scaling_params;
    const models::SampleType m_model_type;

    // A flag to warn only once if the basecall model and read SampleType differ
    std::atomic<bool> m_log_once_inconsistent_read_model{true};
};

}  // namespace dorado
