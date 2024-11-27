#pragma once

#include "decode/Decoder.h"
#include "utils/stats.h"

#include <string>
#include <tuple>
#include <vector>

namespace at {
class Tensor;
}

namespace dorado::basecall {

struct CRFModelConfig;
bool is_duplex_model(const CRFModelConfig &model_config);

class ModelRunnerBase {
public:
    virtual ~ModelRunnerBase() = default;
    virtual void accept_chunk(int chunk_idx, const at::Tensor &chunk) = 0;
    virtual std::vector<decode::DecodedChunk> call_chunks(int num_chunks) = 0;
    virtual const CRFModelConfig &config() const = 0;
    virtual size_t chunk_size() const = 0;
    virtual size_t batch_size() const = 0;

    // Timeout is short for simplex, longer for duplex which gets a subset of reads.
    // Note that these values are overridden for CUDA basecalling.
    virtual std::pair<int, int> batch_timeouts_ms() const {
        return is_duplex_model(config()) ? std::make_pair(5000, 5000) : std::make_pair(100, 100);
    }

    virtual bool is_low_latency() const { return false; }
    virtual void terminate() = 0;
    virtual void restart() = 0;
    virtual std::string get_name() const = 0;
    virtual stats::NamedStats sample_stats() const = 0;
};

using RunnerPtr = std::unique_ptr<ModelRunnerBase>;
enum class PipelineType { simplex_low_latency, simplex, duplex };

}  // namespace dorado::basecall
