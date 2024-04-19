#include "basecall/nn/TxModel.h"

#include "basecall/CRFModelConfig.h"
#include "basecall/nn/CRFModel.h"
#include "utils/gpu_profiling.h"

#include <ATen/Functions.h>
#include <ATen/TensorIndexing.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <spdlog/spdlog.h>
#include <torch/nn.h>
#include <torch/nn/functional/padding.h>
#include <torch/nn/options/padding.h>
#include <torch/serialize.h>
#include <torch/types.h>
#include <torch/version.h>
#if TORCH_VERSION_MAJOR >= 2
#include <ATen/ops/scaled_dot_product_attention.h>
#endif

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace dorado::basecall {

namespace nn {

using namespace torch::nn;
namespace Idx = torch::indexing;
using Slice = torch::indexing::Slice;

torch::Tensor scaled_dot_product_attention_naive(const torch::Tensor &q,
                                                 const torch::Tensor &k,
                                                 const torch::Tensor &v,
                                                 const torch::Tensor &mask) {
    auto matmul_qk = torch::matmul(q, k.transpose(-2, -1));

    auto d_k = k.size(-1);
    matmul_qk = matmul_qk / std::sqrt(d_k);

    if (mask.defined()) {
        matmul_qk = matmul_qk + (mask.logical_not() * -1e9);
    }

    auto weights = torch::softmax(matmul_qk, -1);
    return torch::matmul(weights, v);
}

RMSNormImpl::RMSNormImpl(int hidden_size_) : hidden_size(hidden_size_) {
    weight = at::ones({hidden_size});
    register_parameter("weight", weight, false);
}

at::Tensor RMSNormImpl::forward(at::Tensor x) {
    at::Tensor rstd = torch::rsqrt(x.square().mean(-1, true).add_(eps));
    x.mul_(rstd).mul_(weight);
    return x;
}

GatedMLPImpl::GatedMLPImpl(int in_features, int hidden_features) {
    fc1 = register_module("fc1",
                          Linear(LinearOptions(in_features, 2 * hidden_features).bias(false)));
    fc2 = register_module("fc2", Linear(LinearOptions(hidden_features, in_features).bias(false)));
};

at::Tensor GatedMLPImpl::forward(const at::Tensor &x) {
    const at::Tensor fc1_ = fc1(x);
    const std::vector<at::Tensor> chunks = fc1_.chunk(2, -1);
    const at::Tensor &y = chunks[0];
    const at::Tensor &gate = chunks[1];
    at::Tensor out = fc2(functional::silu(gate).mul_(y));
    return out;
}

RotaryEmbeddingImpl::RotaryEmbeddingImpl(int dim_,
                                         float theta_,
                                         int max_seq_len_,
                                         const at::TensorOptions &options_)
        : dim(dim_), max_seq_len(max_seq_len_), theta(theta_), options(options_) {
    // To maintain precision we use float32 here
    const at::Tensor inv_freq =
            torch::pow(theta, torch::arange(0, dim, 2, options) / dim).reciprocal();

    // freqs.shape := {max_seq_len, 1, 1, dim/2}
    const at::Tensor freqs =
            torch::arange(max_seq_len, options).reshape({max_seq_len, 1, 1, 1}) * inv_freq;

    register_buffer("cos_freqs", torch::cos(freqs).to(options));
    register_buffer("sin_freqs", torch::sin(freqs).to(options));
};

at::Tensor RotaryEmbeddingImpl::forward(at::Tensor &qkv) {
    assert_forward_dims(qkv);
    const int64_t seq_len = qkv.size(1);

    auto buffers = named_buffers();
    const at::Tensor cos_buf = buffers["cos_freqs"].narrow(0, 0, seq_len);
    const at::Tensor sin_buf = buffers["sin_freqs"].narrow(0, 0, seq_len);

    using Slices = std::vector<at::indexing::TensorIndex>;
    const Slices evens = {at::indexing::Ellipsis, at::indexing::Slice(at::indexing::None, 2),
                          at::indexing::Slice(), at::indexing::Slice(at::indexing::None, dim / 2)};
    const Slices odds = {at::indexing::Ellipsis, at::indexing::Slice(at::indexing::None, 2),
                         at::indexing::Slice(), at::indexing::Slice(dim / 2, dim)};

    at::Tensor qk_evens = qkv.index(evens).clone();
    at::Tensor qk_odds = qkv.index(odds);

    qkv.index_put_(evens, cos_buf * qk_evens - sin_buf * qk_odds);
    qkv.index_put_(odds, sin_buf * qk_evens + cos_buf * qk_odds);

    return qkv;
}

void RotaryEmbeddingImpl::assert_forward_dims(const at::Tensor &qkv) const {
    // Expected shape: N, seq_len, 3, nhead, head_dim
    const int64_t seq_len = qkv.size(1);
    const int64_t three = qkv.size(2);
    const int64_t head_dim = qkv.size(4);

    bool has_error = false;
    if (seq_len > max_seq_len) {
        has_error = true;
        spdlog::error(
                "RotE - maximum sequence length exceeded (len:{} > max:{}) - "
                "Your chunksize may be too large",
                seq_len, max_seq_len);
    }
    if (three != 3) {
        has_error = true;
        spdlog::error("RotE - expected constant size:3 at dim:2 found:{}", three);
    }
    if (head_dim != dim) {
        has_error = true;
        spdlog::error("RotE - expected head_dim size:{} at dim:4 found:{}", dim, head_dim);
    }
    if (has_error) {
        throw std::runtime_error("RotE - input dimensions invalid");
    }
}

MultiHeadAttentionImpl::MultiHeadAttentionImpl(int d_model_,
                                               int nhead_,
                                               bool qkv_bias_,
                                               bool out_bias_,
                                               const std::pair<int, int> &attn_window_,
                                               const at::TensorOptions &options_)
        : d_model(d_model_),
          nhead(nhead_),
          head_dim(d_model_ / nhead_),
          attn_window(attn_window_),
          options(options_) {
    wqkv = register_module("wqkv", Linear(LinearOptions(d_model, 3 * d_model).bias(qkv_bias_)));
    out_proj = register_module("out_proj", Linear(LinearOptions(d_model, d_model).bias(out_bias_)));
    const float theta = 10000.0f;
    const int64_t max_seq_len = 2000;
    rotary_emb =
            register_module("rotary_emb", RotaryEmbedding(head_dim, theta, max_seq_len, options));
};

at::Tensor MultiHeadAttentionImpl::build_attn_window_mask(const int64_t size) const {
    const auto [win_upper, win_lower] = attn_window;
    at::Tensor mask = at::ones({size, size}, options.device());
    mask.triu_(-win_upper).tril_(win_lower);
    mask = mask.to(at::kBool);
    return mask;
};

at::Tensor MultiHeadAttentionImpl::forward(at::Tensor x) {
    const int64_t N = x.size(0);
    const int64_t T = x.size(1);
    const int64_t C = x.size(2);

    at::Tensor qkv;
    at::Tensor attn_output;
    {
        utils::ScopedProfileRange spr("QKV", 2);
        // in_feat=512, out_feat=1536 (3*in), nhead=8, head_dim=64=(512/8), dim_ff=2048
        qkv = wqkv(x).view({N, T, 3, nhead, head_dim});
    }
    {
        utils::ScopedProfileRange spr("ROTE", 2);
        qkv = rotary_emb(qkv);
    }
    {
        utils::ScopedProfileRange spr("MEA", 2);
        // NT3HD -> N3HTD -> N[1]HTD
        const auto qkv_ = qkv.permute({0, 2, 3, 1, 4}).chunk(3, 1);
        auto attn_window_mask = build_attn_window_mask(T);

#if TORCH_VERSION_MAJOR < 2
        attn_output =
                scaled_dot_product_attention_naive(qkv_[0], qkv_[1], qkv_[2], attn_window_mask);
#else
        attn_output = at::scaled_dot_product_attention(qkv_[0], qkv_[1], qkv_[2], attn_window_mask);
#endif
        attn_output = attn_output.permute({0, 1, 3, 2, 4}).reshape({N, T, C});
    }
    {
        utils::ScopedProfileRange spr("OUTP", 2);
        x = out_proj(attn_output);
    }
    return x;
};

TxEncoderImpl::TxEncoderImpl(const tx::TxEncoderParams &params, const at::TensorOptions &options) {
    self_attn = register_module("self_attn", MultiHeadAttention(params.d_model, params.nhead, false,
                                                                true, params.attn_window, options));
    ff = register_module("ff", GatedMLP(params.d_model, params.dim_feedforward));
    norm1 = register_module("norm1", RMSNorm(params.d_model));
    norm2 = register_module("norm2", RMSNorm(params.d_model));

    const at::Tensor deepnorm_alpha = at::tensor(params.deepnorm_alpha);
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

TxEncoderStackImpl::TxEncoderStackImpl(const basecall::CRFModelConfig &config,
                                       const at::TensorOptions &options) {
    const auto &tx_enc_params = config.tx->tx;
    stack = Sequential();
    for (int i = 0; i < tx_enc_params.depth; ++i) {
        stack->push_back(register_module("transformer_encoder" + std::to_string(i),
                                         TxEncoder(tx_enc_params, options)));
    }
};

LinearUpsampleImpl::LinearUpsampleImpl(const tx::EncoderUpsampleParams &params)
        : scale_factor(params.scale_factor) {
    linear = register_module(
            "linear",
            Linear(LinearOptions(params.d_model, scale_factor * params.d_model).bias(true)));
};

at::Tensor LinearUpsampleImpl::forward(const at::Tensor &x) {
    const int64_t N = x.size(0);
    const int64_t T = x.size(1);
    const int64_t C = x.size(2);
    at::Tensor out = linear(x).reshape({N, scale_factor * T, C});
    return out;
};

LinearScaledCRFImpl::LinearScaledCRFImpl(const tx::CRFEncoderParams &params) {
    m_params = params;
    linear = register_module(
            "linear", Linear(LinearOptions(m_params.insize, m_params.outsize()).bias(false)));
};

at::Tensor LinearScaledCRFImpl::forward(const at::Tensor &x) { return linear(x) * m_params.scale; }

TxModelImpl::TxModelImpl(const basecall::CRFModelConfig &config, const at::TensorOptions &options)
        : m_options(options) {
    convs = register_module("convs", basecall::nn::ConvStack(config.convs));
    tx_encoder = register_module("transformer_encoder", TxEncoderStack(config, m_options));
    tx_decoder = register_module("transformer_decoder", LinearUpsample(config.tx->upsample));
    crf = register_module("crf", LinearScaledCRF(config.tx->crf));
}

at::Tensor TxModelImpl::forward(const at::Tensor &x) {
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
