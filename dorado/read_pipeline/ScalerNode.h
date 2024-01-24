#pragma once

#include "basecall/CRFModelConfig.h"
#include "read_pipeline/MessageSink.h"
#include "utils/stats.h"

#include <string>

namespace dorado {

class ScalerNode : public MessageSink {
public:
    ScalerNode(const basecall::SignalNormalisationParams& config,
               basecall::SampleType model_type,
               bool trim_adapter,
               int num_worker_threads,
               size_t max_reads);
    ~ScalerNode() { stop_input_processing(); }
    std::string get_name() const override { return "ScalerNode"; }
    stats::NamedStats sample_stats() const override { return stats::from_obj(m_work_queue); }
    void terminate(const FlushOptions&) override { stop_input_processing(); }
    void restart() override { start_input_processing(&ScalerNode::input_thread_fn, this); }

private:
    void input_thread_fn();

    const basecall::SignalNormalisationParams m_scaling_params;
    const basecall::SampleType m_model_type;
    const bool m_trim_adapter;
};

}  // namespace dorado
