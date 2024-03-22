#include "basecall/nn/TxModel.h"

#include "basecall/CRFModelConfig.h"
#include "basecall/nn/CRFModel.h"
#include "utils/gpu_profiling.h"

#include <torch/nn.h>

#include <vector>

namespace dorado {

namespace nn {

using namespace dorado::basecall::tx;
using namespace torch::nn;
namespace Idx = torch::indexing;
using Slice = torch::indexing::Slice;

SwiGLUImpl::SwiGLUImpl(int in_features, int hidden_features) {
    w12 = register_module("w12",
                          Linear(LinearOptions(in_features, 2 * hidden_features).bias(false)));
    w3 = register_module("w3", Linear(LinearOptions(hidden_features, in_features).bias(false)));

    const std::vector<at::Tensor> w12_chunks = w12->weight.chunk(2, 0);
    const at::Tensor &w1 = w12_chunks[0];
    const at::Tensor &w2 = w12_chunks[1];

    register_buffer("w1", w1.detach());
    register_buffer("w2", w2.detach());
};

at::Tensor SwiGLUImpl::forward(at::Tensor x) {
    const at::Tensor x1 = functional::linear(x, named_buffers()["w1"]);
    const at::Tensor x2 = functional::linear(x, named_buffers()["w2"]);
    const at::Tensor y = functional::silu(x1).mul(x2);
    const at::Tensor out = w3(y);
    return out;
}

RotaryEmbeddingImpl::RotaryEmbeddingImpl(int dim, int theta, int max_seq_len)
        : dim(dim), theta(theta), max_seq_len(max_seq_len) {
    const at::Tensor domain = torch::full({1}, theta, torch::kFloat32)
                                      .pow(torch::arange(0, dim, 2)
                                                   .index({Slice{Idx::None, dim / 2}})
                                                   .to(torch::kFloat32)
                                                   .div(dim))
                                      .reciprocal();

    const at::Tensor freqs = torch::arange(max_seq_len).reshape({max_seq_len, 1, 1}) * domain;

    register_buffer("cos_freqs", torch::cos(freqs));
    register_buffer("sin_freqs", torch::sin(freqs));
};

at::Tensor RotaryEmbeddingImpl::forward(at::Tensor x) {
    const long int seq_len = x.size(1);
    // const long int head_dim = x.size(3);

    at::Tensor out = x.clone();

    const at::Tensor cos_buf = named_buffers()["cos_freqs"].index({Slice(Idx::None, seq_len)});
    const at::Tensor sin_buf = named_buffers()["sin_freqs"].index({Slice(Idx::None, seq_len)});

    const at::Tensor evens = x.index({Idx::Ellipsis, Slice{Idx::None, dim, 2}});
    const at::Tensor odds = x.index({Idx::Ellipsis, Slice{1, dim, 2}});

    out.index({Idx::Ellipsis, Slice{Idx::None, dim, 2}}) = (cos_buf * evens) - (sin_buf * odds);
    out.index({Idx::Ellipsis, Slice{1, dim, 2}}) = (sin_buf * evens) + (cos_buf * odds);

    return out;
}

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

MultiHeadAttentionImpl::MultiHeadAttentionImpl(int d_model, int nhead, bool qkv_bias)
        : d_model(d_model), nhead(nhead), head_dim(d_model / nhead) {
    in_proj =
            register_module("in_proj", Linear(LinearOptions(d_model, 3 * d_model).bias(qkv_bias)));
    out_proj = register_module("out_proj", Linear(LinearOptions(d_model, d_model)));
    rotary_emb = register_module("rotary_emb", RotaryEmbedding(head_dim / 2, 10000, 1000));
};

at::Tensor MultiHeadAttentionImpl::forward(at::Tensor x) {
    const long int N = x.size(0);
    const long int T = x.size(1);
    const long int C = x.size(2);

    std::vector<at::Tensor> qkv;
    at::Tensor attn_output;
    {
        utils::ScopedProfileRange spr_qkv("QKV", 2);
        qkv = in_proj(x).view({N, T, nhead, 3 * head_dim}).chunk(3, -1);
    }
    {
        utils::ScopedProfileRange spr_rote("ROTE", 2);
        qkv[0] = rotary_emb(qkv[0]);
        qkv[1] = rotary_emb(qkv[1]);
    }
    {
        utils::ScopedProfileRange spr_rote("MEA", 2);
        ;
        attn_output = at::scaled_dot_product_attention(qkv[0].permute({0, 2, 1, 3}),
                                                       qkv[1].permute({0, 2, 1, 3}),
                                                       qkv[2].permute({0, 2, 1, 3}));
    }
    {
        utils::ScopedProfileRange spr_rote("OUTP", 2);
        x = out_proj(attn_output.permute({0, 2, 1, 3}).reshape({N, T, C}));
    }
    return x;
};

TxEncoderImpl::TxEncoderImpl(const TxEncoderParams &params) {
    self_attn =
            register_module("self_attn", MultiHeadAttention(params.d_model, params.nhead, false));
    ff = register_module("ff", SwiGLU(params.d_model, params.dim_feedforward()));
    norm1 = register_module("norm1", LayerNorm(LayerNormOptions({params.d_model})));
    norm2 = register_module("norm2", LayerNorm(LayerNormOptions({params.d_model})));

    const at::Tensor deepnorm_alpha = at::tensor(params.deepnorm_alpha());
    register_buffer("deepnorm_alpha", deepnorm_alpha);
};

at::Tensor TxEncoderImpl::forward(at::Tensor x) {
    at::Tensor attn, f;
    {
        utils::ScopedProfileRange spr_rote("MHE", 2);
        attn = self_attn(x);
    }
    {
        utils::ScopedProfileRange spr_rote("LNORM1", 2);
        x = norm1((x * named_buffers()["deepnorm_alpha"]) + attn);
    }
    {
        utils::ScopedProfileRange spr_rote("FF", 2);
        f = ff(x);
    }
    {
        utils::ScopedProfileRange spr_rote("LNORM2", 2);
        x = norm2((x * named_buffers()["deepnorm_alpha"]) + f);
    }
    return x;
}

TxEncoderStackImpl::TxEncoderStackImpl(const TxEncoderParams &params) {
    stack = Sequential();
    for (int i = 0; i < params.depth; ++i) {
        stack->push_back(register_module("tx" + std::to_string(i), TxEncoder(params)));
    }
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
    tx_encoder = register_module("transformer_encoder", TxEncoderStack(tx_enc_params));
    tx_decoder = register_module("transformer_decoder", LinearUpsample(tx_enc_params));
    crf = register_module("crf", LinearScaledCRF(tx_enc_params.d_model, 4096, false));
}

// void load_state_dict(const std::vector<at::Tensor> &weights) {
//     utils::load_state_dict(*this, weights);
// }

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

}  // namespace dorado
