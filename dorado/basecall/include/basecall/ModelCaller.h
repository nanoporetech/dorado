#pragma once

#include "DecodedChunk.h"
#include "utils/stats.h"
#include "utils/types.h"

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace at {
class Tensor;
}

namespace dorado::config {
struct BasecallModelConfig;
}

namespace dorado::basecall {

class ModelCaller {
public:
    class RunnerContext {
    public:
        virtual ~RunnerContext() = default;
        virtual void accept_chunk(int chunk_idx, const at::Tensor& chunk) = 0;
        virtual std::vector<decode::DecodedChunk> call_chunks(int num_chunks) = 0;
        virtual size_t chunk_size() const = 0;
        virtual size_t batch_size() const = 0;
        virtual bool variable_chunk_sizes() const { return false; }

    protected:
        RunnerContext() = default;
    };

    virtual ~ModelCaller() = default;
    virtual std::unique_ptr<RunnerContext> create_runner_context(size_t batch_dims_idx) = 0;
    virtual size_t num_batch_dims() const = 0;
    virtual const config::BasecallModelConfig& config() const = 0;
    virtual std::shared_ptr<ModBaseInfo> modbase_info() const = 0;
    virtual std::pair<int, int> batch_timeouts_ms() const;
    virtual bool is_low_latency() const { return false; }
    virtual void terminate() = 0;
    virtual void restart() = 0;
    virtual std::string get_name() const = 0;
    virtual stats::NamedStats sample_stats() const = 0;

protected:
    ModelCaller() = default;
};

}  // namespace dorado::basecall
