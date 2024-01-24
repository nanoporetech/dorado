#pragma once

#include "decode/Decoder.h"
#include "utils/stats.h"

#include <string>
#include <vector>

namespace at {
class Tensor;
}

namespace dorado::basecall {

struct CRFModelConfig;

class ModelRunnerBase {
public:
    virtual ~ModelRunnerBase() = default;
    virtual void accept_chunk(int chunk_idx, const at::Tensor &chunk) = 0;
    virtual std::vector<decode::DecodedChunk> call_chunks(int num_chunks) = 0;
    virtual const CRFModelConfig &config() const = 0;
    virtual size_t model_stride() const = 0;
    virtual size_t chunk_size() const = 0;
    virtual size_t batch_size() const = 0;
    virtual void terminate() = 0;
    virtual void restart() = 0;
    virtual std::string get_name() const = 0;
    virtual stats::NamedStats sample_stats() const = 0;
};

using RunnerPtr = std::unique_ptr<ModelRunnerBase>;

}  // namespace dorado::basecall
