#include "model_latent_space_lstm.h"

#include "torch_utils/tensor_utils.h"

#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace dorado::secondary {

namespace {

enum class ActivationType {
    RELU,
};

torch::nn::Sequential make_1d_conv_layers(const std::vector<int32_t>& kernel_sizes,
                                          int32_t num_in_features,
                                          const std::vector<int32_t>& channels,
                                          const bool use_batch_norm,
                                          const ActivationType activation) {
    if (std::size(kernel_sizes) != std::size(channels)) {
        throw std::invalid_argument("channels and kernel_sizes must have the same size");
    }

    for (const int32_t k : kernel_sizes) {
        if ((k % 2) == 0) {
            throw std::invalid_argument(
                    "Kernel sizes must be odd for equal and symmetric padding. Given: k = " +
                    std::to_string(k));
        }
    }

    torch::nn::Sequential layers;
    for (size_t i = 0; i < std::size(kernel_sizes); ++i) {
        const int32_t k = kernel_sizes[i];
        const int32_t c = channels[i];

        layers->push_back(torch::nn::Conv1d(
                torch::nn::Conv1dOptions(num_in_features, c, k).padding((k - 1) / 2)));

        if (activation == ActivationType::RELU) {
            layers->push_back(torch::nn::ReLU());
        } else {
            throw std::invalid_argument("Unsupported activation function!");
        }

        if (use_batch_norm) {
            layers->push_back(torch::nn::BatchNorm1d(c));
        }

        num_in_features = c;
    }

    return layers;
}

}  // namespace

ReadLevelConvImpl::ReadLevelConvImpl(const int32_t num_in_features,
                                     const int32_t out_dim,
                                     const std::vector<int32_t>& kernel_sizes,
                                     const std::vector<int32_t>& channel_dims,
                                     bool use_batch_norm)
        : m_convs{make_1d_conv_layers(kernel_sizes,
                                      num_in_features,
                                      channel_dims,
                                      use_batch_norm,
                                      ActivationType::RELU)},
          m_expansion_layer{torch::nn::Linear(channel_dims.back(), out_dim)}

{
    if (std::size(kernel_sizes) != std::size(channel_dims)) {
        throw std::runtime_error(
                "Wrong number of values in kernel_sizes or channel_dims. Got: "
                "kernel_sizes.size() = " +
                std::to_string(std::size(kernel_sizes)) +
                ", channel_dims.size() = " + std::to_string(std::size(channel_dims)));
    }

    register_module("convs", m_convs);
    register_module("expansion_layer", m_expansion_layer);
}

torch::Tensor ReadLevelConvImpl::forward(torch::Tensor x) { return m_convs->forward(std::move(x)); }

torch::Tensor MeanPoolerImpl::forward(const torch::Tensor& x,
                                      const torch::Tensor& non_empty_position_mask) {
    const auto read_depths = non_empty_position_mask.sum(-1).unsqueeze(-1).unsqueeze(-1);
    const auto mask = non_empty_position_mask.unsqueeze(-1).unsqueeze(-1);
    return (x * mask).sum(1) / read_depths;
}

ReversibleLSTM::ReversibleLSTM(const int32_t input_size,
                               const int32_t hidden_size,
                               const bool batch_first,
                               const bool reverse)
        : m_lstm(torch::nn::LSTMOptions(input_size, hidden_size).batch_first(batch_first)),
          m_batch_first{batch_first},
          m_reverse(reverse) {
    register_module("lstm", m_lstm);
}

torch::Tensor ReversibleLSTM::forward(torch::Tensor x) {
    const int32_t flip_dim = m_batch_first ? 1 : 0;
    if (m_reverse) {
        x = x.flip(flip_dim);
    }
    auto output = std::get<0>(m_lstm->forward(x));
    if (m_reverse) {
        output = output.flip(flip_dim);
    }
    return output;
}

ModelLatentSpaceLSTM::ModelLatentSpaceLSTM(const int32_t num_classes,
                                           const int32_t lstm_size,
                                           const int32_t cnn_size,
                                           const std::vector<int32_t>& kernel_sizes,
                                           const std::string& pooler_type,
                                           const bool use_dwells,
                                           const int32_t bases_alphabet_size,
                                           const int32_t bases_embedding_size,
                                           const bool bidirectional)
        : m_num_classes{num_classes},
          m_lstm_size{lstm_size},
          m_cnn_size{cnn_size},
          m_kernel_sizes{kernel_sizes},
          m_pooler_type{pooler_type},
          m_use_dwells{use_dwells},
          m_bases_alphabet_size{bases_alphabet_size},
          m_bases_embedding_size{bases_embedding_size},
          m_bidirectional{bidirectional},

          m_base_embedder(
                  torch::nn::EmbeddingOptions(m_bases_alphabet_size, m_bases_embedding_size)),
          m_strand_embedder(torch::nn::EmbeddingOptions(3, m_bases_embedding_size)),
          m_read_level_conv(m_bases_embedding_size + (m_use_dwells ? 2 : 1),
                            m_lstm_size,
                            m_kernel_sizes,
                            std::vector<int32_t>(std::size(m_kernel_sizes), m_cnn_size),
                            true),
          m_pre_pool_expansion_layer(m_cnn_size, m_lstm_size),
          m_pooler(MeanPooler()),
          m_lstm_bidir(torch::nn::LSTMOptions(m_lstm_size, m_lstm_size)
                               .num_layers(2)
                               .batch_first(true)
                               .bidirectional(m_bidirectional)),
          m_lstm_unidir(),
          m_linear((1 + m_bidirectional) * m_lstm_size, m_num_classes) {
    if (m_bidirectional) {
        m_lstm_bidir = torch::nn::LSTM(torch::nn::LSTMOptions(m_lstm_size, m_lstm_size)
                                               .num_layers(2)
                                               .batch_first(true)
                                               .bidirectional(m_bidirectional));
    } else {
        for (int32_t i = 0; i < 4; ++i) {
            m_lstm_unidir->push_back(ReversibleLSTM(m_lstm_size, m_lstm_size, true, !(i % 2)));
        }
    }

    if (m_pooler_type != "mean") {
        throw std::runtime_error("Pooler " + m_pooler_type + " not implemented yet.");
    }

    register_module("base_embedder", m_base_embedder);
    register_module("strand_embedder", m_strand_embedder);
    register_module("read_level_conv", m_read_level_conv);
    register_module("pre_pool_expansion_layer", m_pre_pool_expansion_layer);
    register_module("pooler", m_pooler);
    if (m_bidirectional) {
        register_module("lstm", m_lstm_bidir);
    } else {
        register_module("lstm", m_lstm_unidir);
    }
    register_module("linear", m_linear);
}

torch::Tensor ModelLatentSpaceLSTM::forward(torch::Tensor x) {
    // Non-const because it will be moved later. Needs to be placed here because x changes.
    auto non_empty_position_mask = (x.sum({1, -1}) != 0);

    auto bases_embedding =
            m_base_embedder->forward(x.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                              torch::indexing::Slice(), 0})
                                             .to(torch::kLong));
    auto strand_embedding =
            m_strand_embedder->forward(x.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                                torch::indexing::Slice(), 2})
                                               .to(torch::kLong) +
                                       1);
    auto scaled_q_scores = (x.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                     torch::indexing::Slice(), 1}) /
                                    25 -
                            1)
                                   .unsqueeze(-1);

    if (m_use_dwells) {
        if (x.sizes().back() != 5) {
            throw std::runtime_error(
                    "If using dwells, x must have 5 features/read/position. Shape of x: " +
                    utils::tensor_shape_as_string(x));
        }
        auto dwells = x.index({torch::indexing::Slice(), torch::indexing::Slice(),
                               torch::indexing::Slice(), 4})
                              .unsqueeze(-1);
        x = torch::cat(
                {bases_embedding + strand_embedding, std::move(scaled_q_scores), std::move(dwells)},
                -1);
    } else {
        x = torch::cat({bases_embedding + strand_embedding, std::move(scaled_q_scores)}, -1);
    }

    x = x.permute({0, 2, 3, 1});

    // The sizes() returns a torch::IntArrayRef
    const auto b = at::size(x, 0);
    const auto d = at::size(x, 1);
    const auto p = at::size(x, 3);

    x = x.flatten(0, 1);
    x = m_read_level_conv->forward(std::move(x));
    x = x.permute({0, 2, 1});
    x = m_pre_pool_expansion_layer->forward(x);
    x = x.view({b, d, p, m_lstm_size});
    x = m_pooler->forward(x, non_empty_position_mask);
    x = m_bidirectional ? std::get<0>(m_lstm_bidir->forward(x)) : m_lstm_unidir->forward(x);
    x = m_linear->forward(x);

    return x;
}

}  // namespace dorado::secondary
