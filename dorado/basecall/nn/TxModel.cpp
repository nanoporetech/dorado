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
#include <c10/core/TensorOptions.h>
#include <torch/nn.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace dorado::basecall {

namespace nn {

using namespace dorado::basecall::tx;
using namespace torch::nn;
namespace Idx = torch::indexing;
using Slice = torch::indexing::Slice;

std::string shape(const at::Tensor &t, const std::string &name) {
    std::string str = name + ".shape()={";
    const auto &sz = t.sizes();
    for (size_t i = 0; i < sz.size(); ++i) {
        if (i != 0) {
            str += ", ";
        }
        str += std::to_string(sz[i]);
    }
    str += "}";
    return str;
}

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
    // TODO: can max_seq_len 1000 -> attn_window.size()?
    rotary_emb = register_module("rotary_emb",
                                 RotaryEmbedding(head_dim / 2, 10000, attn_window_mask.size(0)));
};

at::Tensor MultiHeadAttentionImpl::forward(at::Tensor x) {
    spdlog::debug(shape(x, "MHA.x"));
    const long int N = x.size(0);
    const long int T = x.size(1);
    const long int C = x.size(2);

    std::vector<at::Tensor> qkv;
    at::Tensor attn_output;
    {
        utils::ScopedProfileRange spr("QKV", 2);
        // in_feat=512, out_feat=1536 (3*in), nhead=8, head_dim=64=(512/8), dim_ff=2048

        qkv = wqkv(x).view({N, T, 3, nhead, head_dim}).permute({0, 1, 3, 4, 2}).chunk(3, -1);
        // qkv = wqkv(x).view({N, T, nhead, 3 * head_dim}).chunk(3, -1);

        // N, T, 8, 192 -> N, T, 8, 64
        // qkv[0].size := N, T, nhead(8), head_dim(64)
        spdlog::debug(shape(qkv[0].squeeze(), "MHA.qkv[0]"));
    }
    {
        utils::ScopedProfileRange spr("ROTE", 2);
        qkv[0] = rotary_emb(qkv[0].squeeze());
        qkv[1] = rotary_emb(qkv[1].squeeze());
        qkv[2] = qkv[2].squeeze();
    }
    {
        // if (qkv[0].size(1) != attn_window_mask.size(0)) {
        //     spdlog::error("attn_window_mask size error: attn={} qkv[0].size(1)={}",
        //                   attn_window_mask.size(0), qkv[0].size(1));
        //     spdlog::error("N: {} T: {} C: {}", N, T, C);
        //     const auto &sz = qkv[0].sizes();
        //     for (size_t i = 0; i < sz.size(); ++i) {
        //         spdlog::error("qkv[0].size({})={}", i, sz[i]);
        //     }
        // }
        utils::ScopedProfileRange spr("MEA", 2);

        spdlog::debug(shape(qkv[0], "MHA.qkv[0].post"));
        spdlog::debug(shape(qkv[0].permute({0, 2, 1, 3}), "MHA.qkv[0].permute({0, 2, 1, 3})"));
        spdlog::debug(shape(attn_window_mask, "MHA.attn_mask"));
        attn_output = at::scaled_dot_product_attention(
                // permute := N, nhead(8), T, head_dim(64)
                qkv[0].permute({0, 2, 1, 3}), qkv[1].permute({0, 2, 1, 3}),
                qkv[2].permute({0, 2, 1, 3}), attn_window_mask);
    }
    {
        utils::ScopedProfileRange spr("OUTP", 2);
        x = out_proj(attn_output.permute({0, 2, 1, 3}).reshape({N, T, C}));
        spdlog::debug(shape(x, "MHA.out_proj"));
    }
    return x;
};

TxEncoderImpl::TxEncoderImpl(const TxEncoderParams &params, const at::Tensor &attn_window_mask_)
        : attn_window_mask(attn_window_mask_) {
    self_attn = register_module("self_attn", MultiHeadAttention(params.d_model, params.nhead, false,
                                                                true, attn_window_mask));
    ff = register_module("ff", GatedMLP(params.d_model, params.dim_feedforward));
    norm1 = register_module("norm1", RMSNorm(params.d_model));
    norm2 = register_module("norm2", RMSNorm(params.d_model));

    const at::Tensor deepnorm_alpha = at::tensor(params.deepnorm_alpha());
    register_buffer("deepnorm_alpha", deepnorm_alpha);
};

at::Tensor TxEncoderImpl::forward(at::Tensor x) {
    spdlog::debug(shape(x, "TxEncoder.x"));
    at::Tensor attn, f;
    {
        utils::ScopedProfileRange spr("MHE", 2);
        attn = self_attn(x);
        spdlog::debug(shape(attn, "TxEncoder.attn"));
    }
    {
        utils::ScopedProfileRange spr("LNORM1", 2);
        x = norm1(attn + (x * named_buffers()["deepnorm_alpha"]));
        spdlog::debug(shape(x, "TxEncoder.norm1"));
    }
    {
        utils::ScopedProfileRange spr("FF", 2);
        f = ff(x);
        spdlog::debug(shape(f, "TxEncoder.ff"));
    }
    {
        utils::ScopedProfileRange spr("LNORM2", 2);
        x = norm2(f + (x * named_buffers()["deepnorm_alpha"]));
        spdlog::debug(shape(x, "TxEncoder.norm2"));
    }
    return x;
}

TxEncoderStackImpl::TxEncoderStackImpl(const basecall::CRFModelConfig &config,
                                       const at::TensorOptions &options)
        : attn_window_mask(build_attn_window_mask(config, options)) {
    const auto &tx_enc_params = config.tx->tx;
    stack = Sequential();
    for (int i = 0; i < tx_enc_params.depth; ++i) {
        stack->push_back(register_module("transformer_encoder" + std::to_string(i),
                                         TxEncoder(tx_enc_params, attn_window_mask)));
    }
};

at::Tensor TxEncoderStackImpl::build_attn_window_mask(const basecall::CRFModelConfig &config,
                                                      const at::TensorOptions &options) const {
    const int size =
            config.basecaller.chunksize / (config.stride * config.tx->upsample.scale_factor);
    const auto [win_upper, win_lower] = config.tx->tx.attn_window;

    at::Tensor mask = at::triu(at::ones({size, size}), -win_upper);
    mask *= at::tril(mask, win_lower);
    mask = mask.to(at::kBool).to(options.device());

    spdlog::debug(shape(mask, "TxEncoderStack.mask"));
    return mask;
};

LinearUpsampleImpl::LinearUpsampleImpl(const EncoderUpsampleParams &params)
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
}

TxModelImpl::TxModelImpl(const basecall::CRFModelConfig &config, const at::TensorOptions &options)
        : m_options(options) {
    const auto conv_params = config.convs;
    convs = register_module("convs", basecall::nn::ConvStack(conv_params));

    const auto tx_enc_params = config.tx->tx;
    tx_encoder = register_module("transformer_encoder", TxEncoderStack(config, m_options));
    tx_decoder = register_module("transformer_decoder", LinearUpsample(config.tx->upsample));
    crf = register_module("crf", LinearScaledCRF(tx_enc_params.d_model, 4096, false));
}

at::Tensor TxModelImpl::forward(at::Tensor x) {
    spdlog::debug(shape(x, "TxModel.x"));

    at::Tensor h;
    {
        utils::ScopedProfileRange spr("Conv", 1);
        h = convs->forward(x);
        spdlog::debug(shape(h, "TxModel.convs.h"));
    }
    {
        utils::ScopedProfileRange spr("TransEnc", 1);
        h = tx_encoder(h);
        spdlog::debug(shape(h, "TxModel.tx_encoder.h"));
    }
    {
        utils::ScopedProfileRange spr("TransDec", 1);
        h = tx_decoder(h);
        spdlog::debug(shape(h, "TxModel.tx_decoder.h"));
    }
    {
        utils::ScopedProfileRange spr("CRF", 1);
        h = crf(h);
        spdlog::debug(shape(h, "TxModel.crf.h"));
    }
    return h;
}

}  // namespace nn

}  // namespace dorado::basecall
