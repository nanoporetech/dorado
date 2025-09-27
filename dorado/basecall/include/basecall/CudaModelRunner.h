#pragma once

#include "ModelRunnerBase.h"

#include <ATen/core/TensorBody.h>
#include <c10/cuda/CUDAStream.h>

#include <atomic>
#include <filesystem>
#include <memory>
#include <tuple>
#include <vector>

namespace dorado::basecall {
class CudaCaller;

class CudaModelRunner final : public ModelRunnerBase {
public:
    explicit CudaModelRunner(std::shared_ptr<CudaCaller> caller, size_t batch_dims_idx);
    void accept_chunk(int chunk_idx, const at::Tensor& chunk) final;
    std::vector<decode::DecodedChunk> call_chunks(int num_chunks) final;
    const config::BasecallModelConfig& config() const final;
    size_t chunk_size() const final;
    size_t batch_size() const final;
    bool variable_chunk_sizes() const final;
    std::pair<int, int> batch_timeouts_ms() const final;
    bool is_low_latency() const final;
    void terminate() final;
    void restart() final;
    std::string get_name() const final;
    stats::NamedStats sample_stats() const final;

private:
    std::shared_ptr<CudaCaller> m_caller;
    at::Tensor m_input;
    at::Tensor m_output;
    at::Tensor m_aux;
    std::size_t m_batch_size;
    std::size_t m_chunk_size;
    std::vector<std::int32_t> m_chunk_sizes;
    std::size_t m_chunk_offset{0};
    c10::cuda::CUDAStream m_stream;

    // Performance monitoring stats.
    std::atomic<int64_t> m_num_batches_called = 0;
};

}  // namespace dorado::basecall
