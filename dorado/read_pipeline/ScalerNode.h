#pragma once

#include "basecall/CRFModelConfig.h"
#include "read_pipeline/MessageSink.h"
#include "torch_utils/trim_rapid_adapter.h"
#include "utils/stats.h"

#include <atomic>
#include <string>

namespace dorado {

class ScalerNode : public MessageSink {
public:
    ScalerNode(const basecall::SignalNormalisationParams& config,
               models::SampleType model_type,
               bool trim_rna_adapter,
               const utils::rapid::Settings& m_rapid_settings,
               int num_worker_threads,
               size_t max_reads);
    ~ScalerNode() { stop_input_processing(); }
    std::string get_name() const override { return "ScalerNode"; }
    stats::NamedStats sample_stats() const override { return stats::from_obj(m_work_queue); }
    void terminate(const FlushOptions&) override { stop_input_processing(); }
    void restart() override {
        start_input_processing([this] { input_thread_fn(); }, "scaler_node");
    }

private:
    void input_thread_fn();

    const basecall::SignalNormalisationParams m_scaling_params;
    const models::SampleType m_model_type;
    const bool m_trim_rna_adapter;
    const utils::rapid::Settings m_rapid_settings;

    // A flag to warn only once if the basecall model and read SampleType differ
    std::atomic<bool> m_log_once_inconsistent_read_model{true};
};

}  // namespace dorado
