#pragma once

#include "ModelRunner.h"

#include <torch/torch.h>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace dorado {

class MetalCaller;

std::shared_ptr<MetalCaller> create_metal_caller(const CRFModelConfig& model_config,
                                                 int chunk_size,
                                                 int batch_size);

class MetalModelRunner final : public ModelRunnerBase {
public:
    explicit MetalModelRunner(std::shared_ptr<MetalCaller> caller);
    void accept_chunk(int chunk_idx, const torch::Tensor& chunk) final;
    std::vector<DecodedChunk> call_chunks(int num_chunks) final;
    size_t model_stride() const final;
    size_t chunk_size() const final;
    size_t batch_size() const final;
    void terminate() final;
    std::string get_name() const final { return "MetalModelRunner"; }
    stats::NamedStats sample_stats() const final;

private:
    std::shared_ptr<MetalCaller> m_caller;
    torch::Tensor m_input;

    // Performance monitoring stats.
    std::atomic<int64_t> m_num_batches_called = 0;
};

}  // namespace dorado
