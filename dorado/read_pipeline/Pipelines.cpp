#include "Pipelines.h"

#include "BasecallerNode.h"
#include "DuplexSplitNode.h"
#include "ModBaseCallerNode.h"
#include "PairingNode.h"
#include "ScalerNode.h"
#include "StereoDuplexEncoderNode.h"
#include "nn/CRFModelConfig.h"
#include "nn/ModBaseRunner.h"
#include "nn/ModelRunner.h"

#include <spdlog/spdlog.h>

namespace dorado::pipelines {

void create_simplex_pipeline(PipelineDescriptor& pipeline_desc,
                             std::vector<dorado::Runner>&& runners,
                             std::vector<std::unique_ptr<dorado::ModBaseRunner>>&& modbase_runners,
                             size_t overlap,
                             uint32_t mean_qscore_start_pos,
                             int scaler_node_threads,
                             int modbase_node_threads,
                             NodeHandle sink_node_handle,
                             NodeHandle source_node_handle) {
    const auto& model_config = runners.front()->config();
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
                {}, std::move(modbase_runners), modbase_node_threads, model_stride);
    }

    const int kBatchTimeoutMS = 100;
    std::string model_name =
            std::filesystem::canonical(model_config.model_path).filename().string();

    auto basecaller_node = pipeline_desc.add_node<BasecallerNode>(
            {}, std::move(runners), overlap, kBatchTimeoutMS, model_name, 1000, "BasecallerNode",
            false, mean_qscore_start_pos);

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
        pipeline_desc.add_node_sink(last_node_handle, sink_node_handle);
    }
}

void create_stereo_duplex_pipeline(PipelineDescriptor& pipeline_desc,
                                   std::vector<dorado::Runner>&& runners,
                                   std::vector<dorado::Runner>&& stereo_runners,
                                   size_t overlap,
                                   uint32_t mean_qscore_start_pos,
                                   int scaler_node_threads,
                                   int splitter_node_threads,
                                   PairingParameters pairing_parameters,
                                   NodeHandle sink_node_handle,
                                   NodeHandle source_node_handle) {
    const auto& model_config = runners.front()->config();
    const auto& stereo_model_config = stereo_runners.front()->config();
    std::string model_name =
            std::filesystem::canonical(model_config.model_path).filename().string();
    auto stereo_model_name =
            std::filesystem::canonical(stereo_model_config.model_path).filename().string();
    auto duplex_rg_name = std::string(model_name + "_" + stereo_model_name);
    auto stereo_model_stride = stereo_runners.front()->model_stride();
    auto adjusted_stereo_overlap = (overlap / stereo_model_stride) * stereo_model_stride;
    const int kStereoBatchTimeoutMS = 5000;

    auto stereo_basecaller_node = pipeline_desc.add_node<BasecallerNode>(
            {}, std::move(stereo_runners), adjusted_stereo_overlap, kStereoBatchTimeoutMS,
            duplex_rg_name, 1000, "StereoBasecallerNode", true, mean_qscore_start_pos);

    auto simplex_model_stride = runners.front()->model_stride();
    auto stereo_node = pipeline_desc.add_node<StereoDuplexEncoderNode>({stereo_basecaller_node},
                                                                       simplex_model_stride);

    auto pairing_node =
            std::holds_alternative<ReadOrder>(pairing_parameters)
                    ? pipeline_desc.add_node<PairingNode>({stereo_node},
                                                          std::get<ReadOrder>(pairing_parameters),
                                                          std::thread::hardware_concurrency())
                    : pipeline_desc.add_node<PairingNode>(
                              {stereo_node}, std::move(std::get<std::map<std::string, std::string>>(
                                                     pairing_parameters)));

    // Create a duplex split node with the given settings and number of devices.
    // If splitter_settings.enabled is set to false, the splitter node will act
    // as a passthrough, meaning it won't perform any splitting operations and
    // will just pass data through.
    DuplexSplitSettings splitter_settings;
    auto splitter_node = pipeline_desc.add_node<DuplexSplitNode>({pairing_node}, splitter_settings,
                                                                 splitter_node_threads);

    auto adjusted_simplex_overlap = (overlap / simplex_model_stride) * simplex_model_stride;

    const int kSimplexBatchTimeoutMS = 100;
    auto basecaller_node = pipeline_desc.add_node<BasecallerNode>(
            {splitter_node}, std::move(runners), adjusted_simplex_overlap, kSimplexBatchTimeoutMS,
            model_name, 1000, "BasecallerNode", true, mean_qscore_start_pos);

    auto scaler_node = pipeline_desc.add_node<ScalerNode>(
            {basecaller_node}, model_config.signal_norm_params, scaler_node_threads);

    // if we've been provided a source node, connect it to the start of our pipeline
    if (source_node_handle != PipelineDescriptor::InvalidNodeHandle) {
        pipeline_desc.add_node_sink(source_node_handle, scaler_node);
    }

    // if we've been provided a sink node, connect it to the end of our pipeline
    if (sink_node_handle != PipelineDescriptor::InvalidNodeHandle) {
        pipeline_desc.add_node_sink(stereo_basecaller_node, sink_node_handle);
    }
}

}  // namespace dorado::pipelines
