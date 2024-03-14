#include "pipeline_creation.h"

#include "basecall/CRFModelConfig.h"
#include "basecall/ModelRunnerBase.h"
#include "modbase/ModBaseRunner.h"
#include "read_pipeline/BasecallerNode.h"
#include "read_pipeline/ModBaseCallerNode.h"
#include "read_pipeline/PairingNode.h"
#include "read_pipeline/ReadSplitNode.h"
#include "read_pipeline/ScalerNode.h"
#include "read_pipeline/StereoDuplexEncoderNode.h"
#include "splitter/DuplexReadSplitter.h"
#include "splitter/RNAReadSplitter.h"

#include <spdlog/spdlog.h>

namespace dorado::api {

void create_simplex_pipeline(PipelineDescriptor& pipeline_desc,
                             std::vector<basecall::RunnerPtr>&& runners,
                             std::vector<modbase::RunnerPtr>&& modbase_runners,
                             size_t overlap,
                             uint32_t mean_qscore_start_pos,
                             bool trim_adapter,
                             int scaler_node_threads,
                             bool enable_read_splitter,
                             int splitter_node_threads,
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

    std::string model_name =
            std::filesystem::canonical(model_config.model_path).filename().string();

    const bool is_rna = is_rna_model(model_config);
    NodeHandle first_node_handle = PipelineDescriptor::InvalidNodeHandle;
    NodeHandle last_node_handle = PipelineDescriptor::InvalidNodeHandle;

    NodeHandle current_node_handle = PipelineDescriptor::InvalidNodeHandle;

    // For RNA model, read splitting happens first before any basecalling.
    if (enable_read_splitter && is_rna) {
        splitter::RNASplitSettings rna_splitter_settings;
        auto rna_splitter =
                std::make_unique<const splitter::RNAReadSplitter>(rna_splitter_settings);
        auto rna_splitter_node = pipeline_desc.add_node<ReadSplitNode>({}, std::move(rna_splitter),
                                                                       splitter_node_threads, 1000);
        first_node_handle = rna_splitter_node;
        current_node_handle = rna_splitter_node;
    }

    auto trim_rapid_adapter_settings = utils::rapid::get_settings();
    auto scaler_node = pipeline_desc.add_node<ScalerNode>(
            {}, model_config.signal_norm_params, model_config.sample_type, trim_adapter,
            trim_rapid_adapter_settings, scaler_node_threads, 1000);
    if (current_node_handle != PipelineDescriptor::InvalidNodeHandle) {
        pipeline_desc.add_node_sink(current_node_handle, scaler_node);
    } else {
        first_node_handle = scaler_node;
    }
    current_node_handle = scaler_node;
    auto basecaller_node =
            pipeline_desc.add_node<BasecallerNode>({}, std::move(runners), overlap, model_name,
                                                   1000, "BasecallerNode", mean_qscore_start_pos);
    pipeline_desc.add_node_sink(current_node_handle, basecaller_node);
    current_node_handle = basecaller_node;
    last_node_handle = basecaller_node;

    // For DNA, read splitting happens after basecall.
    if (enable_read_splitter && !is_rna) {
        splitter::DuplexSplitSettings splitter_settings(model_config.signal_norm_params.strategy ==
                                                        basecall::ScalingStrategy::PA);
        splitter_settings.simplex_mode = true;
        auto dna_splitter = std::make_unique<const splitter::DuplexReadSplitter>(splitter_settings);
        auto dna_splitter_node = pipeline_desc.add_node<ReadSplitNode>({}, std::move(dna_splitter),
                                                                       splitter_node_threads, 1000);
        pipeline_desc.add_node_sink(current_node_handle, dna_splitter_node);
        current_node_handle = dna_splitter_node;
        last_node_handle = dna_splitter_node;
    }

    if (!modbase_runners.empty()) {
        auto mod_base_caller_node = pipeline_desc.add_node<ModBaseCallerNode>(
                {}, std::move(modbase_runners), modbase_node_threads, model_stride, 1000);
        pipeline_desc.add_node_sink(current_node_handle, mod_base_caller_node);
        current_node_handle = mod_base_caller_node;
        last_node_handle = mod_base_caller_node;
    }

    // if we've been provided a source node, connect it to the start of our pipeline
    if (source_node_handle != PipelineDescriptor::InvalidNodeHandle) {
        pipeline_desc.add_node_sink(source_node_handle, first_node_handle);
    }

    // if we've been provided a sink node, connect it to the end of our pipeline
    if (sink_node_handle != PipelineDescriptor::InvalidNodeHandle) {
        pipeline_desc.add_node_sink(last_node_handle, sink_node_handle);
    }
}

void create_stereo_duplex_pipeline(PipelineDescriptor& pipeline_desc,
                                   std::vector<basecall::RunnerPtr>&& runners,
                                   std::vector<basecall::RunnerPtr>&& stereo_runners,
                                   std::vector<modbase::RunnerPtr>&& modbase_runners,
                                   size_t overlap,
                                   uint32_t mean_qscore_start_pos,
                                   int scaler_node_threads,
                                   int splitter_node_threads,
                                   int modbase_node_threads,
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

    auto stereo_basecaller_node = pipeline_desc.add_node<BasecallerNode>(
            {}, std::move(stereo_runners), adjusted_stereo_overlap, duplex_rg_name, 1000,
            "StereoBasecallerNode", mean_qscore_start_pos);

    NodeHandle last_node_handle = stereo_basecaller_node;
    if (!modbase_runners.empty()) {
        auto mod_base_caller_node = pipeline_desc.add_node<ModBaseCallerNode>(
                {}, std::move(modbase_runners), modbase_node_threads,
                size_t(runners.front()->model_stride()), 1000);
        pipeline_desc.add_node_sink(stereo_basecaller_node, mod_base_caller_node);
        last_node_handle = mod_base_caller_node;
    }

    auto simplex_model_stride = runners.front()->model_stride();
    auto stereo_node = pipeline_desc.add_node<StereoDuplexEncoderNode>({stereo_basecaller_node},
                                                                       int(simplex_model_stride));

    auto pairing_node =
            std::holds_alternative<DuplexPairingParameters>(pairing_parameters)
                    ? pipeline_desc.add_node<PairingNode>(
                              {stereo_node}, std::get<DuplexPairingParameters>(pairing_parameters),
                              std::thread::hardware_concurrency(), 1000)
                    : pipeline_desc.add_node<PairingNode>(
                              {stereo_node},
                              std::move(std::get<std::map<std::string, std::string>>(
                                      pairing_parameters)),
                              2, 1000);

    // Create a duplex split node with the given settings and number of devices.
    // If splitter_settings.enabled is set to false, the splitter node will act
    // as a passthrough, meaning it won't perform any splitting operations and
    // will just pass data through.
    splitter::DuplexSplitSettings splitter_settings(model_config.signal_norm_params.strategy ==
                                                    basecall::ScalingStrategy::PA);
    auto duplex_splitter = std::make_unique<const splitter::DuplexReadSplitter>(splitter_settings);
    auto splitter_node = pipeline_desc.add_node<ReadSplitNode>(
            {pairing_node}, std::move(duplex_splitter), splitter_node_threads, 1000);

    auto adjusted_simplex_overlap = (overlap / simplex_model_stride) * simplex_model_stride;

    auto basecaller_node = pipeline_desc.add_node<BasecallerNode>(
            {splitter_node}, std::move(runners), adjusted_simplex_overlap, model_name, 1000,
            "BasecallerNode", mean_qscore_start_pos);

    // TODO: Do we want to trim rapid adapters in duplex?

    utils::rapid::Settings trim_rapid_adapter_settings;
    trim_rapid_adapter_settings.active = false;
    auto scaler_node = pipeline_desc.add_node<ScalerNode>(
            {basecaller_node}, model_config.signal_norm_params, basecall::SampleType::DNA, false,
            trim_rapid_adapter_settings, scaler_node_threads, 1000);

    // if we've been provided a source node, connect it to the start of our pipeline
    if (source_node_handle != PipelineDescriptor::InvalidNodeHandle) {
        pipeline_desc.add_node_sink(source_node_handle, scaler_node);
    }

    // if we've been provided a sink node, connect it to the end of our pipeline
    if (sink_node_handle != PipelineDescriptor::InvalidNodeHandle) {
        pipeline_desc.add_node_sink(last_node_handle, sink_node_handle);
    }
}

}  // namespace dorado::api
