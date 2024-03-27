#include "basecall/nn/TxModel.h"

#include "basecall/CRFModelConfig.h"
#include "basecall/nn/CRFModel.h"
#include "spdlog/spdlog.h"
#include "utils/gpu_profiling.h"

#include <ATen/core/TensorBody.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/full.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/sqrt.h>
#include <ATen/ops/tensor.h>
#include <ATen/ops/tril.h>
#include <ATen/ops/triu.h>
#include <c10/core/ScalarType.h>
#include <torch/nn.h>

#include <stdexcept>
#include <vector>

namespace dorado::basecall {

namespace nn {

using namespace dorado::basecall::tx;
using namespace torch::nn;
namespace Idx = torch::indexing;
using Slice = torch::indexing::Slice;

RMSNormImpl::RMSNormImpl(int hidden_size_) : hidden_size(hidden_size_) {
    weight = at::ones({hidden_size});
    register_parameter("weight", weight, false);
}

at::Tensor RMSNormImpl::forward(at::Tensor x) {
    at::Tensor rstd = torch::rsqrt(x.square().mean(-1, true) + eps);
    return x * rstd * weight;
}

GatedMLPImpl::GatedMLPImpl(int in_features, int hidden_features) {
    fc1 = register_module("fc1",
                          Linear(LinearOptions(in_features, 2 * hidden_features).bias(false)));
    fc2 = register_module("fc2", Linear(LinearOptions(hidden_features, in_features).bias(false)));
};

at::Tensor GatedMLPImpl::forward(at::Tensor x) {
    const std::vector<at::Tensor> chunks = fc1(x).chunk(2, -1);
    const at::Tensor &y = chunks[0];
    const at::Tensor &gate = chunks[1];
    return fc2(functional::silu(gate).mul(y));
}

RotaryEmbeddingImpl::RotaryEmbeddingImpl(int dim_, int theta_, int max_seq_len_)
        : dim(dim_), theta(theta_), max_seq_len(max_seq_len_) {
    const at::Tensor inv_freq = torch::full({1}, theta, torch::kFloat32)
                                        .pow(torch::arange(0, dim, 2)
                                                     .index({Slice{Idx::None, dim / 2}})
                                                     .to(torch::kFloat32)
                                                     .div(dim))
                                        .reciprocal();

    const at::Tensor freqs = torch::arange(max_seq_len).reshape({max_seq_len, 1, 1}) * inv_freq;

    register_buffer("cos_freqs", torch::cos(freqs));
    register_buffer("sin_freqs", torch::sin(freqs));
};

at::Tensor RotaryEmbeddingImpl::forward(at::Tensor x) {
    const long int seq_len = x.size(1);

    at::Tensor out = x.clone();

    const at::Tensor cos_buf = named_buffers()["cos_freqs"].index({Slice(Idx::None, seq_len)});
    const at::Tensor sin_buf = named_buffers()["sin_freqs"].index({Slice(Idx::None, seq_len)});

    const at::Tensor evens = x.index({Idx::Ellipsis, Slice{Idx::None, dim, 2}});
    const at::Tensor odds = x.index({Idx::Ellipsis, Slice{1, dim, 2}});

    out.index({Idx::Ellipsis, Slice{Idx::None, dim, 2}}) = (cos_buf * evens) - (sin_buf * odds);
    out.index({Idx::Ellipsis, Slice{1, dim, 2}}) = (sin_buf * evens) + (cos_buf * odds);

    return out;
}

MultiHeadAttentionImpl::MultiHeadAttentionImpl(int d_model_,
                                               int nhead_,
                                               bool qkv_bias_,
                                               bool out_bias_,
                                               const at::Tensor &attn_window_mask_)
        : d_model(d_model_),
          nhead(nhead_),
          head_dim(d_model_ / nhead_),
          attn_window_mask(attn_window_mask_) {
    wqkv = register_module("wqkv", Linear(LinearOptions(d_model, 3 * d_model).bias(qkv_bias_)));
    out_proj = register_module("out_proj", Linear(LinearOptions(d_model, d_model).bias(out_bias_)));
    rotary_emb = register_module("rotary_emb", RotaryEmbedding(head_dim / 2, 10000, 1000));
};

at::Tensor MultiHeadAttentionImpl::forward(at::Tensor x) {
    const long int N = x.size(0);
    const long int T = x.size(1);
    const long int C = x.size(2);

    std::vector<at::Tensor> qkv;
    at::Tensor attn_output;
    {
        utils::ScopedProfileRange spr("QKV", 2);
        // in_feat=512, out_feat=1536 (3*in), nhead=8, head_dim=64=(512/8), dim_ff=2048
        qkv = wqkv(x).view({N, T, nhead, 3 * head_dim}).chunk(3, -1);
        // N, T, 8, 192 -> N, T, 8, 64
        // qkv[0].size := N, T, nhead(8), head_dim(64)
    }
    {
        utils::ScopedProfileRange spr("ROTE", 2);
        qkv[0] = rotary_emb(qkv[0]);
        qkv[1] = rotary_emb(qkv[1]);
    }
    {
        if (qkv[0].size(1) != attn_window_mask.size(0)) {
            spdlog::error("attn_window_mask size error: attn={} qkv[0].size(1)={}",
                          attn_window_mask.size(0), qkv[0].size(1));
        }
        utils::ScopedProfileRange spr("MEA", 2);
        attn_output = at::scaled_dot_product_attention(
                // permute := N, nhead(8), T, head_dim(64)
                qkv[0].permute({0, 2, 1, 3}), qkv[1].permute({0, 2, 1, 3}),
                qkv[2].permute({0, 2, 1, 3}), attn_window_mask);
    }
    {
        utils::ScopedProfileRange spr("OUTP", 2);
        x = out_proj(attn_output.permute({0, 2, 1, 3}).reshape({N, T, C}));
    }
    return x;
};

TxEncoderImpl::TxEncoderImpl(const TxEncoderParams &params, const at::Tensor &attn_window_mask_)
        : attn_window_mask(attn_window_mask_) {
    self_attn = register_module("self_attn", MultiHeadAttention(params.d_model, params.nhead, false,
                                                                true, attn_window_mask));
    ff = register_module("ff", GatedMLP(params.d_model, params.dim_feedforward()));
    norm1 = register_module("norm1", RMSNorm(params.d_model));
    norm2 = register_module("norm2", RMSNorm(params.d_model));

    const at::Tensor deepnorm_alpha = at::tensor(params.deepnorm_alpha());
    register_buffer("deepnorm_alpha", deepnorm_alpha);
};

at::Tensor TxEncoderImpl::forward(at::Tensor x) {
    at::Tensor attn, f;
    {
        utils::ScopedProfileRange spr("MHE", 2);
        attn = self_attn(x);
    }
    {
        utils::ScopedProfileRange spr("LNORM1", 2);
        x = norm1(attn + (x * named_buffers()["deepnorm_alpha"]));
    }
    {
        utils::ScopedProfileRange spr("FF", 2);
        f = ff(x);
    }
    {
        utils::ScopedProfileRange spr("LNORM2", 2);
        x = norm2(f + (x * named_buffers()["deepnorm_alpha"]));
    }
    return x;
}

TxEncoderStackImpl::TxEncoderStackImpl(const basecall::CRFModelConfig &config)
        : attn_window_mask(build_attn_window_mask(config)) {
    const auto &tx_enc_params = config.tx->tx;
    stack = Sequential();
    for (int i = 0; i < tx_enc_params.depth; ++i) {
        stack->push_back(register_module("transformer_encoder" + std::to_string(i),
                                         TxEncoder(tx_enc_params, attn_window_mask)));
    }
};

at::Tensor TxEncoderStackImpl::build_attn_window_mask(
        const basecall::CRFModelConfig &config) const {
    const int size = config.convs.back().size;
    const auto [win_upper, win_lower] = config.tx->tx.attn_window;

    at::Tensor mask = at::triu(at::full(size, 1), -win_upper);
    mask *= at::tril(mask, win_lower);
    mask = mask.to(at::kBool);
    return mask;
};

LinearUpsampleImpl::LinearUpsampleImpl(const TxEncoderParams &params)
        : scale_factor(params.scale_factor) {
    linear = register_module("linear",
                             Linear(LinearOptions(params.d_model, scale_factor * params.d_model)));
};
at::Tensor LinearUpsampleImpl::forward(at::Tensor x) {
    const long int N = x.size(0);
    const long int T = x.size(1);
    const long int C = x.size(2);
    auto out = linear(x).reshape({N, scale_factor * T, C});
    return out;
};

LinearScaledCRFImpl::LinearScaledCRFImpl(int insize, int outsize, bool bias) {
    linear = register_module("linear", Linear(LinearOptions(insize, outsize).bias(bias)));
};

at::Tensor LinearScaledCRFImpl::forward(at::Tensor x) {
    utils::ScopedProfileRange spr("linscale", 2);
    return linear(x) * 5.0f;
    ;
}

TxModelImpl::TxModelImpl(const basecall::CRFModelConfig &config) {
    const auto conv_params = config.convs;
    convs = register_module("convs", basecall::nn::ConvStack(conv_params));

    const auto tx_enc_params = config.tx->tx;
    tx_encoder = register_module("transformer_encoder", TxEncoderStack(config));
    tx_decoder = register_module("transformer_decoder", LinearUpsample(tx_enc_params));
    crf = register_module("crf", LinearScaledCRF(tx_enc_params.d_model, 4096, false));
}

at::Tensor TxModelImpl::forward(at::Tensor x) {
    at::Tensor h;
    {
        utils::ScopedProfileRange spr("Conv", 1);
        h = convs->forward(x);
    }
    {
        utils::ScopedProfileRange spr("TransEnc", 1);
        h = tx_encoder(h);
    }
    {
        utils::ScopedProfileRange spr("TransDec", 1);
        h = tx_decoder(h);
    }
    {
        utils::ScopedProfileRange spr("CRF", 1);
        h = crf(h);
    }
    return h;
}

}  // namespace nn

}  // namespace dorado::basecall
