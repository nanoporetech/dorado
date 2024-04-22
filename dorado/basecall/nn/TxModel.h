#pragma once

#include "basecall/CRFModelConfig.h"
#include "basecall/nn/CRFModel.h"
#include "utils/gpu_profiling.h"
#include "utils/module_utils.h"

#include <c10/core/TensorOptions.h>
#include <torch/nn.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace dorado::basecall {

namespace nn {

torch::Tensor scaled_dot_product_attention_naive(const torch::Tensor &q,
                                                 const torch::Tensor &k,
                                                 const torch::Tensor &v,
                                                 const torch::Tensor &mask);

struct RMSNormImpl : torch::nn::Module {
    RMSNormImpl(int hidden_size_);
    at::Tensor forward(at::Tensor x);

    at::Tensor weight;
    const int hidden_size;
    const float eps{1e-5f};
};

TORCH_MODULE(RMSNorm);

struct GatedMLPImpl : torch::nn::Module {
    GatedMLPImpl(int in_features, int hidden_features);

    at::Tensor forward(const at::Tensor &x);

    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

TORCH_MODULE(GatedMLP);

struct RotaryEmbeddingImpl : torch::nn::Module {
    RotaryEmbeddingImpl(int dim_,
                        float theta_,
                        int max_seq_len_,
                        const at::TensorOptions &options_);

    at::Tensor forward(const at::Tensor &qkv);
    void assert_forward_dims(const at::Tensor &qkv) const;

    const int64_t dim, max_seq_len;
    const float theta;
    const at::TensorOptions options;
};

TORCH_MODULE(RotaryEmbedding);

struct MultiHeadAttentionImpl : torch::nn::Module {
    MultiHeadAttentionImpl(int d_model_,
                           int nhead_,
                           bool qkv_bias_,
                           bool out_bias_,
                           const std::pair<int, int> &attn_window_,
                           const at::TensorOptions &options_);

    at::Tensor forward(at::Tensor x);

    at::Tensor build_attn_window_mask(const int64_t size) const;

    const int d_model, nhead, head_dim;
    const std::pair<int, int> attn_window;
    const at::TensorOptions options;

    torch::nn::Linear wqkv{nullptr}, out_proj{nullptr};
    RotaryEmbedding rotary_emb{nullptr};
};

TORCH_MODULE(MultiHeadAttention);

struct TxEncoderImpl : torch::nn::Module {
    TxEncoderImpl(const basecall::tx::TxEncoderParams &params, const at::TensorOptions &options);

    at::Tensor forward(at::Tensor x);

    MultiHeadAttention self_attn{nullptr};
    GatedMLP ff{nullptr};
    RMSNorm norm1{nullptr}, norm2{nullptr};
};

TORCH_MODULE(TxEncoder);

struct TxEncoderStackImpl : torch::nn::Module {
    TxEncoderStackImpl(const basecall::CRFModelConfig &config, const at::TensorOptions &options);

    at::Tensor forward(const at::Tensor &x) { return stack->forward(x); };
    torch::nn::Sequential stack{nullptr};
};

TORCH_MODULE(TxEncoderStack);

struct LinearUpsampleImpl : torch::nn::Module {
    LinearUpsampleImpl(const basecall::tx::EncoderUpsampleParams &params);

    at::Tensor forward(const at::Tensor &x);
    const int scale_factor;
    torch::nn::Linear linear{nullptr};
};

TORCH_MODULE(LinearUpsample);

struct LinearScaledCRFImpl : torch::nn::Module {
    LinearScaledCRFImpl(const tx::CRFEncoderParams &params);

    at::Tensor forward(const at::Tensor &x);

    torch::nn::Linear linear{nullptr};
    tx::CRFEncoderParams m_params;
};

TORCH_MODULE(LinearScaledCRF);

struct TxModelImpl : torch::nn::Module {
    explicit TxModelImpl(const basecall::CRFModelConfig &config, const at::TensorOptions &options);

    void load_state_dict(const std::vector<at::Tensor> &weights) {
        utils::load_state_dict(*this, weights);
    }

    at::Tensor forward(const at::Tensor &x);

    basecall::nn::ConvStack convs{nullptr};
    TxEncoderStack tx_encoder{nullptr};
    LinearUpsample tx_decoder{nullptr};
    LinearScaledCRF crf{nullptr};

    const at::TensorOptions m_options;
};

TORCH_MODULE(TxModel);

}  // namespace nn

}  // namespace dorado::basecall
