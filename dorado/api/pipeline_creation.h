#pragma once

#include "read_pipeline/ReadPipeline.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace dorado {

namespace basecall {
class ModelRunnerBase;
using RunnerPtr = std::unique_ptr<ModelRunnerBase>;
}  // namespace basecall

namespace modbase {
class ModBaseRunner;
using RunnerPtr = std::unique_ptr<ModBaseRunner>;
}  // namespace modbase

using PairingParameters = std::variant<DuplexPairingParameters, std::map<std::string, std::string>>;

namespace api {

/// Create a simplex basecall pipeline description
/// If source_node_handle is valid, set this to be the source of the simplex pipeline
/// If sink_node_handle is valid, set this to be the sink of the simplex pipeline
void create_simplex_pipeline(PipelineDescriptor& pipeline_desc,
                             std::vector<basecall::RunnerPtr>&& runners,
                             std::vector<modbase::RunnerPtr>&& modbase_runners,
                             uint32_t mean_qscore_start_pos,
                             int scaler_node_threads,
                             bool enable_read_splitter,
                             int splitter_node_threads,
                             int modbase_threads,
                             NodeHandle sink_node_handle,
                             NodeHandle source_node_handle);

/// Create a duplex basecall pipeline description
/// If source_node_handle is valid, set this to be the source of the simplex pipeline
/// If sink_node_handle is valid, set this to be the sink of the simplex pipeline
void create_stereo_duplex_pipeline(PipelineDescriptor& pipeline_desc,
                                   std::vector<basecall::RunnerPtr>&& runners,
                                   std::vector<basecall::RunnerPtr>&& stereo_runners,
                                   std::vector<modbase::RunnerPtr>&& modbase_runners,
                                   uint32_t mean_qscore_start_pos,
                                   int scaler_node_threads,
                                   int splitter_node_threads,
                                   int modbase_node_threads,
                                   PairingParameters pairing_parameters,
                                   NodeHandle sink_node_handle,
                                   NodeHandle source_node_handle);

}  // namespace api

}  // namespace dorado
