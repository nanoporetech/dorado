#pragma once

#include "model_torch_base.h"
#include "polish/polish_utils.h"

#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace dorado::polisher {

// Class for ReadLevelConv
class ReadLevelConvImpl : public torch::nn::Module {
public:
    ReadLevelConvImpl(const int32_t num_in_features,             // 5
                      const int32_t out_dim,                     // 128
                      const std::vector<int32_t>& kernel_sizes,  // [1, 17]
                      const std::vector<int32_t>& channel_dims,
                      bool use_batch_norm  // True
    );

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Sequential m_convs;
    torch::nn::Linear m_expansion_layer;
};
TORCH_MODULE(ReadLevelConv);

class MeanPoolerImpl : public torch::nn::Module {
public:
    torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& non_empty_position_mask);
};
TORCH_MODULE(MeanPooler);

class ReversibleLSTM : public torch::nn::Module {
public:
    ReversibleLSTM(const int32_t input_size,
                   const int32_t hidden_size,
                   const bool batch_first,
                   const bool reverse);

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::LSTM m_lstm;
    bool m_batch_first = false;
    bool m_reverse = false;
};

class ModelLatentSpaceLSTM : public ModelTorchBase {
public:
    ModelLatentSpaceLSTM(const int32_t num_classes,
                         const int32_t lstm_size,
                         const int32_t cnn_size,
                         const std::vector<int32_t>& kernel_sizes,
                         const std::string& pooler_type,
                         const bool use_dwells,
                         const int32_t bases_alphabet_size,
                         const int32_t bases_embedding_size,
                         const bool bidirectional);

    torch::Tensor forward(torch::Tensor x);

private:
    // Model parameters.
    int32_t m_num_classes = 5;
    int32_t m_lstm_size = 128;
    int32_t m_cnn_size = 128;
    std::vector<int32_t> m_kernel_sizes = {1, 17};
    std::string m_pooler_type = "mean";
    bool m_use_dwells = false;
    int32_t m_bases_alphabet_size = 6;
    int32_t m_bases_embedding_size = 6;
    bool m_bidirectional = true;
    bool m_normalise = true;

    // Layers.
    torch::nn::Embedding m_base_embedder;
    torch::nn::Embedding m_strand_embedder;
    ReadLevelConv m_read_level_conv;
    torch::nn::Linear m_pre_pool_expansion_layer;
    MeanPooler m_pooler{nullptr};
    torch::nn::LSTM m_lstm_bidir;
    torch::nn::Sequential m_lstm_unidir{nullptr};
    torch::nn::Linear m_linear;
};

}  // namespace dorado::polisher
