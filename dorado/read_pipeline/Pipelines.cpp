#include "Pipelines.h"

#include "BasecallerNode.h"
#include "ModBaseCallerNode.h"
#include "ScalerNode.h"
#include "nn/CRFModel.h"
#include "nn/ModBaseRunner.h"
#include "nn/ModelRunner.h"

#include <spdlog/spdlog.h>

namespace dorado::pipelines {

void create_simplex_pipeline(PipelineDescriptor& pipeline_desc,
                             const CRFModelConfig& model_config,
                             std::vector<dorado::Runner>&& runners,
                             std::vector<std::unique_ptr<dorado::ModBaseRunner>>&& modbase_runners,
                             size_t overlap,
                             int scaler_node_threads,
                             int modbase_threads,
                             NodeHandle sink_node_handle,
                             NodeHandle source_node_handle) {
    auto model_stride = runners.front()->model_stride();
    auto adjusted_overlap = (overlap / model_stride) * model_stride;
    if (overlap != adjusted_overlap) {
        spdlog::debug("- adjusted overlap to match model stride: {} -> {}", overlap,
                      adjusted_overlap);
        overlap = adjusted_overlap;
    }

    auto mod_base_caller_node = PipelineDescriptor::InvalidNodeHandle;
    if (!modbase_runners.empty()) {
        mod_base_caller_node = pipeline_desc.add_node<ModBaseCallerNode>(
                {}, std::move(modbase_runners), modbase_threads, model_stride);
    }

    const int kBatchTimeoutMS = 100;
    std::string model_name =
            std::filesystem::canonical(model_config.model_path).filename().string();

    auto basecaller_node = pipeline_desc.add_node<BasecallerNode>(
            {}, std::move(runners), overlap, kBatchTimeoutMS, model_name, 1000, "BasecallerNode",
            false, get_model_mean_qscore_start_pos(model_config));

    NodeHandle last_node_handle = PipelineDescriptor::InvalidNodeHandle;
    if (mod_base_caller_node != PipelineDescriptor::InvalidNodeHandle) {
        pipeline_desc.add_node_sink(basecaller_node, mod_base_caller_node);
        last_node_handle = mod_base_caller_node;
    } else {
        last_node_handle = basecaller_node;
    }

    auto scaler_node = pipeline_desc.add_node<ScalerNode>(
            {basecaller_node}, model_config.signal_norm_params, scaler_node_threads);

    // if we've been provided a source node, connect it to the start of our pipeline
    if (source_node_handle != PipelineDescriptor::InvalidNodeHandle) {
        pipeline_desc.add_node_sink(source_node_handle, scaler_node);
    }

    // if we've been provided a sink node, connect it to the end of our pipeline
    if (sink_node_handle != PipelineDescriptor::InvalidNodeHandle) {
        if (mod_base_caller_node != PipelineDescriptor::InvalidNodeHandle) {
            pipeline_desc.add_node_sink(mod_base_caller_node, sink_node_handle);
        } else {
            pipeline_desc.add_node_sink(basecaller_node, sink_node_handle);
        }
    }
}

}  // namespace dorado::pipelines
