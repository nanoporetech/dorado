#pragma once

#include <string>
#include <torch/torch.h>
#include "../decode/Decoder.h"

class ModelRunner {
    public:
        virtual void accept_chunk(int chunk_idx, at::Tensor slice) = 0;
        virtual std::vector<DecodedChunk> call_chunks(int num_chunks) = 0;
    protected:
        std::string m_device;
        torch::Tensor m_input;
        torch::TensorOptions m_options;
        DecoderOptions m_decoder_options;
        torch::nn::ModuleHolder<torch::nn::AnyModule> m_module{nullptr};
};
