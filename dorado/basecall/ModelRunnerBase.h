#pragma once

#include "decode/Decoder.h"
#include "utils/stats.h"

#include <string>
#include <utility>
#include <vector>

namespace at {
class Tensor;
}

namespace dorado::config {
struct BasecallModelConfig;
}  // namespace dorado::config

namespace dorado::basecall {

class ModelRunnerBase {
public:
    virtual ~ModelRunnerBase() = default;
    virtual void accept_chunk(int chunk_idx, const at::Tensor &chunk) = 0;
    virtual std::vector<decode::DecodedChunk> call_chunks(int num_chunks) = 0;
    virtual const config::BasecallModelConfig &config() const = 0;
    virtual size_t chunk_size() const = 0;
    virtual size_t batch_size() const = 0;

    // Timeout is short for simplex, longer for duplex which gets a subset of reads.
    // Note that these values are overridden for CUDA basecalling.
    virtual std::pair<int, int> batch_timeouts_ms() const;
    virtual bool is_low_latency() const { return false; }
    virtual void terminate() = 0;
    virtual void restart() = 0;
    virtual std::string get_name() const = 0;
    virtual stats::NamedStats sample_stats() const = 0;
};

using RunnerPtr = std::unique_ptr<ModelRunnerBase>;
enum class PipelineType { simplex_low_latency, simplex, duplex };

struct BasecallerCreationParams {
    const config::BasecallModelConfig &model_config;
    const std::string &device;
    float memory_limit_fraction;
    PipelineType pipeline_type;
    float batch_size_time_penalty;
    bool run_batchsize_benchmarks;
    bool emit_batchsize_benchmarks;
};

}  // namespace dorado::basecall
