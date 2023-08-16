#pragma once

#include "ModelRunner.h"
#include "nn/CRFModel.h"

#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include <filesystem>
#include <memory>
#include <vector>

namespace dorado {

struct CRFModelConfig;
class CudaCaller;

std::shared_ptr<CudaCaller> create_cuda_caller(const CRFModelConfig& model_config,
                                               int chunk_size,
                                               int batch_size,
                                               const std::string& device,
                                               float memory_limit_fraction = 1.f,
                                               bool exclusive_gpu_access = true);

class CudaModelRunner : public ModelRunnerBase {
public:
    explicit CudaModelRunner(std::shared_ptr<CudaCaller> caller);
    void accept_chunk(int chunk_idx, const torch::Tensor& chunk) final;
    std::vector<DecodedChunk> call_chunks(int num_chunks) final;
    const CRFModelConfig& config() const final;
    size_t model_stride() const final;
    size_t chunk_size() const final;
    size_t batch_size() const final;
    void terminate() final;
    void restart() final;
    std::string get_name() const final;
    stats::NamedStats sample_stats() const final;

private:
    std::shared_ptr<CudaCaller> m_caller;
    c10::cuda::CUDAStream m_stream;
    torch::Tensor m_input;
    torch::Tensor m_output;

    // Performance monitoring stats.
    std::atomic<int64_t> m_num_batches_called = 0;
};

}  // namespace dorado
