#pragma once

#include <torch/torch.h>
#include "ModelRunner.h"

class MetalCaller;

std::shared_ptr<MetalCaller> create_metal_caller(const std::string& model_path, int chunk_size, int batch_size);

class MetalModelRunner : public ModelRunnerBase {
public:
    MetalModelRunner(std::shared_ptr<MetalCaller> caller, int chunk_size, int batch_size);
    void accept_chunk(int chunk_idx, at::Tensor slice) final;
    std::vector<DecodedChunk> call_chunks(int num_chunks) final;
private:
    std::shared_ptr<MetalCaller> m_caller;
    torch::Tensor m_input;
};
