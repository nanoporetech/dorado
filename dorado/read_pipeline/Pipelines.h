#pragma once

#include "ReadPipeline.h"
#include "utils/types.h"

#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace dorado {

class CRFModelConfig;
class ModelRunnerBase;
class ModBaseRunner;

using Runner = class std::shared_ptr<dorado::ModelRunnerBase>;
using PairingParameters = std::variant<ReadOrder, std::map<std::string, std::string>>;

namespace pipelines {

/// Create a simplex basecall pipeline description
/// If source_node_handle is valid, set this to be the source of the simplex pipeline
/// If sink_node_handle is valid, set this to be the sink of the simplex pipeline
void create_simplex_pipeline(PipelineDescriptor& pipeline_desc,
                             const CRFModelConfig& model_config,
                             std::vector<dorado::Runner>&& runners,
                             std::vector<std::unique_ptr<dorado::ModBaseRunner>>&& modbase_runners,
                             size_t overlap,
                             int scaler_node_threads,
                             int modbase_threads,
                             NodeHandle sink_node_handle = PipelineDescriptor::InvalidNodeHandle,
                             NodeHandle source_node_handle = PipelineDescriptor::InvalidNodeHandle);

/// Create a duplex basecall pipeline description
/// If source_node_handle is valid, set this to be the source of the simplex pipeline
/// If sink_node_handle is valid, set this to be the sink of the simplex pipeline
void create_stereo_duplex_pipeline(
        PipelineDescriptor& pipeline_desc,
        const CRFModelConfig& model_config,
        const CRFModelConfig& stereo_model_config,
        std::vector<dorado::Runner>&& runners,
        std::vector<dorado::Runner>&& stereo_runners,
        size_t overlap,
        int scaler_node_threads,
        int splitter_node_threads,
        PairingParameters pairing_parameters,
        DuplexSplitSettings splitter_settings,
        NodeHandle sink_node_handle = PipelineDescriptor::InvalidNodeHandle,
        NodeHandle source_node_handle = PipelineDescriptor::InvalidNodeHandle);

}  // namespace pipelines

}  // namespace dorado
