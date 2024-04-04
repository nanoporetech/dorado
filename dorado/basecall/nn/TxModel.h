#pragma once

#include "basecall/CRFModelConfig.h"
#include "basecall/nn/CRFModel.h"
#include "utils/gpu_profiling.h"
#include "utils/module_utils.h"

#include <torch/nn.h>

#include <cstdint>
#include <vector>

namespace dorado::basecall {

namespace nn {

struct RMSNormImpl : torch::nn::Module {
    RMSNormImpl(int layer_num_, int hidden_size_);
    at::Tensor forward(at::Tensor x);

    at::Tensor weight;
    const int layer_num, hidden_size;
    const float eps{1e-5f};
};

TORCH_MODULE(RMSNorm);

struct GatedMLPImpl : torch::nn::Module {
    GatedMLPImpl(int layer_num_, int in_features, int hidden_features);

    at::Tensor forward(at::Tensor x);

    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    const int layer_num;
};

TORCH_MODULE(GatedMLP);

struct RotaryEmbeddingImpl : torch::nn::Module {
    RotaryEmbeddingImpl(int layer_num_, int dim_, int theta_, int max_seq_len_);

    at::Tensor forward(at::Tensor x);

    const int layer_num, dim, theta, max_seq_len;
};

TORCH_MODULE(RotaryEmbedding);

struct MultiHeadAttentionImpl : torch::nn::Module {
    MultiHeadAttentionImpl(int layer_num_,
                           int d_model_,
                           int nhead_,
                           bool qkv_bias_,
                           bool out_bias_,
                           const at::Tensor &attn_window_mask_);

    at::Tensor forward(at::Tensor x);

    const int layer_num, d_model, nhead, head_dim;
    const at::Tensor &attn_window_mask;

    torch::nn::Linear wqkv{nullptr}, out_proj{nullptr};
    RotaryEmbedding rotary_emb{nullptr};
};

TORCH_MODULE(MultiHeadAttention);

struct TxEncoderImpl : torch::nn::Module {
    TxEncoderImpl(const int layer_num_,
                  const basecall::tx::TxEncoderParams &params,
                  const at::Tensor &attn_window_mask_);

    at::Tensor forward(at::Tensor x);

    const int layer_num;
    const at::Tensor &attn_window_mask;
    MultiHeadAttention self_attn{nullptr};
    GatedMLP ff{nullptr};
    RMSNorm norm1{nullptr}, norm2{nullptr};
};

TORCH_MODULE(TxEncoder);

struct TxEncoderStackImpl : torch::nn::Module {
    TxEncoderStackImpl(const basecall::CRFModelConfig &config, const at::TensorOptions &options);

    at::Tensor forward(at::Tensor x) { return stack->forward(x); };
    at::Tensor build_attn_window_mask(const basecall::CRFModelConfig &config,
                                      const at::TensorOptions &options) const;

    const at::Tensor attn_window_mask;
    torch::nn::Sequential stack{nullptr};
};

TORCH_MODULE(TxEncoderStack);

struct LinearUpsampleImpl : torch::nn::Module {
    LinearUpsampleImpl(const basecall::tx::EncoderUpsampleParams &params);

    at::Tensor forward(at::Tensor x);

    const int scale_factor;
    torch::nn::Linear linear{nullptr};
};

TORCH_MODULE(LinearUpsample);

struct LinearScaledCRFImpl : torch::nn::Module {
    LinearScaledCRFImpl(int insize, int outsize, bool bias);

    at::Tensor forward(at::Tensor x);

    torch::nn::Linear linear{nullptr};
};

TORCH_MODULE(LinearScaledCRF);

struct TxModelImpl : torch::nn::Module {
    explicit TxModelImpl(const basecall::CRFModelConfig &config, const at::TensorOptions &options);

    void load_state_dict(const std::vector<at::Tensor> &weights) {
        utils::load_state_dict(*this, weights, {});
    }

    at::Tensor forward(at::Tensor x);

    basecall::nn::ConvStack convs{nullptr};
    TxEncoderStack tx_encoder{nullptr};
    LinearUpsample tx_decoder{nullptr};
    LinearScaledCRF crf{nullptr};

    const at::TensorOptions m_options;
};

TORCH_MODULE(TxModel);

}  // namespace nn

}  // namespace dorado::basecall
