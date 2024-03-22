#pragma once

#include "basecall/CRFModelConfig.h"
#include "basecall/nn/CRFModel.h"
#include "utils/gpu_profiling.h"

#include <torch/nn.h>

#include <vector>

namespace dorado {

namespace nn {

struct SwiGLUImpl : torch::nn::Module {
    SwiGLUImpl(int in_features, int hidden_features);

    at::Tensor forward(at::Tensor x);

    torch::nn::Linear w12{nullptr}, w3{nullptr};
};

TORCH_MODULE(SwiGLU);

struct RotaryEmbeddingImpl : torch::nn::Module {
    RotaryEmbeddingImpl(int dim, int theta, int max_seq_len);

    at::Tensor forward(at::Tensor x);

    const int dim, theta, max_seq_len;
};

TORCH_MODULE(RotaryEmbedding);

struct LinearUpsampleImpl : torch::nn::Module {
    LinearUpsampleImpl(const basecall::tx::TxEncoderParams &params);

    at::Tensor forward(at::Tensor x);

    const int scale_factor;
    torch::nn::Linear linear{nullptr};
};

TORCH_MODULE(LinearUpsample);

struct MultiHeadAttentionImpl : torch::nn::Module {
    MultiHeadAttentionImpl(int d_model, int nhead, bool qkv_bias);

    at::Tensor forward(at::Tensor x);
    const int d_model, nhead, head_dim;
    torch::nn::Linear in_proj{nullptr}, out_proj{nullptr};
    RotaryEmbedding rotary_emb{nullptr};
};

TORCH_MODULE(MultiHeadAttention);

struct TxEncoderImpl : torch::nn::Module {
    TxEncoderImpl(const basecall::tx::TxEncoderParams &params);

    at::Tensor forward(at::Tensor x);

    MultiHeadAttention self_attn{nullptr};
    SwiGLU ff{nullptr};
    torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
};

TORCH_MODULE(TxEncoder);

struct TxEncoderStackImpl : torch::nn::Module {
    TxEncoderStackImpl(const basecall::tx::TxEncoderParams &params);

    at::Tensor forward(at::Tensor x) { return stack->forward(x); };

    torch::nn::Sequential stack{nullptr};
};

TORCH_MODULE(TxEncoderStack);

struct LinearScaledCRFImpl : torch::nn::Module {
    LinearScaledCRFImpl(int insize, int outsize, bool bias);

    at::Tensor forward(at::Tensor x);

    torch::nn::Linear linear{nullptr};
};

TORCH_MODULE(LinearScaledCRF);

struct TxModelImpl : torch::nn::Module {
    explicit TxModelImpl(const basecall::CRFModelConfig &config);

    // void load_state_dict(const std::vector<at::Tensor> &weights) {
    //     utils::load_state_dict(*this, weights);
    // }

    at::Tensor forward(at::Tensor x);

    basecall::nn::ConvStack convs{nullptr};
    TxEncoderStack tx_encoder{nullptr};
    LinearUpsample tx_decoder{nullptr};
    LinearScaledCRF crf{nullptr};
};

TORCH_MODULE(TxModel);

}  // namespace nn

}  // namespace dorado
