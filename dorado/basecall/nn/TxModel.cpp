#include "basecall/nn/TxModel.h"

#include "basecall/CRFModelConfig.h"
#include "basecall/nn/CRFModel.h"
#include "torch_utils/gpu_profiling.h"
#include "utils/dev_utils.h"
#include "utils/math_utils.h"

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

#include <cmath>
#if TORCH_VERSION_MAJOR >= 2
#include <ATen/ops/scaled_dot_product_attention.h>
#endif

#if DORADO_CUDA_BUILD
extern "C" {
#include "koi.h"
}

static bool koi_can_use_cutlass() {
    cudaDeviceProp *prop = at::cuda::getCurrentDeviceProperties();
    return ((prop->major == 8 || prop->major == 9) && prop->minor == 0);
}
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

GatedMLPImpl::GatedMLPImpl(int in_features_, int hidden_features_)
        : in_features(in_features_), hidden_features(hidden_features_) {
    fc1 = register_module("fc1",
                          Linear(LinearOptions(in_features, 2 * hidden_features).bias(false)));
    fc2 = register_module("fc2", Linear(LinearOptions(hidden_features, in_features).bias(false)));
};

at::Tensor GatedMLPImpl::forward(const at::Tensor &x) {
    at::Tensor t;
#if DORADO_CUDA_BUILD && !defined(DORADO_TX2)
    auto use_koi_swiglu = x.is_cuda() && utils::get_dev_opt<bool>("use_koi_swiglu", true) &&
                          koi_can_use_cutlass();
    if (use_koi_swiglu) {
        utils::ScopedProfileRange spr("FC1+SILU", 3);
        auto N = x.size(0);
        auto T = x.size(1);
        if (!features_interleaved) {
            fc1->weight = fc1->weight.view({2, hidden_features, -1})
                                  .transpose(0, 1)
                                  .contiguous()
                                  .view({-1, in_features});
            features_interleaved = true;
        }
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        t = torch::empty({N, T, hidden_features}, x.options());
        int res = host_linear_swiglu_f16(stream, int(N * T), 2 * hidden_features, in_features,
                                         x.data_ptr(), fc1->weight.data_ptr(), t.data_ptr());
        if (res != KOI_SUCCESS) {
            throw std::runtime_error("Koi SwiGLU failed.");
        }
    } else
#endif
    {
        {
            utils::ScopedProfileRange spr("FC1", 3);
            t = fc1(x);
        }
        {
            utils::ScopedProfileRange spr("SILU", 3);
            const auto chunks = t.chunk(2, -1);
            const auto &y = chunks[0];
            const auto &gate = chunks[1];
            t = functional::silu(gate).mul_(y);
        }
    }
    {
        utils::ScopedProfileRange spr("FC2", 3);
        return fc2(t);
    }
}

RotaryEmbeddingImpl::RotaryEmbeddingImpl(int dim_,
                                         float theta_,
                                         int max_seq_len_,
                                         const at::TensorOptions &options_)
        : dim(dim_), max_seq_len(max_seq_len_), theta(theta_), options(options_) {
    const at::Tensor inv_freq = get_inv_freqs();

    // freqs.shape := {max_seq_len, 1, 1, dim/2}
    const at::Tensor freqs =
            torch::arange(max_seq_len, options).reshape({max_seq_len, 1, 1, 1}) * inv_freq;

    register_buffer("cos_freqs", torch::cos(freqs).to(options));
    register_buffer("sin_freqs", torch::sin(freqs).to(options));
};

at::Tensor RotaryEmbeddingImpl::get_inv_freqs() const {
    // Torch2.0 does not have support for ATen::pow in the MPS(apple) backend.
    // Use a vector and std::pow from cmath instead and cast to a tensor

    // Equivalent to:
    // const at::Tensor inv_freq =
    //         torch::pow(theta, torch::arange(0, dim, 2, options) / dim).reciprocal();

    // TODO: Remove when updating to torch2.1+
    std::vector<double> vec;
    vec.reserve(dim / 2);
    for (float i = 0; i < dim; i += 2) {
        vec.push_back(std::pow(static_cast<double>(theta), static_cast<double>(i / (float)dim)));
    }
    at::Tensor inv_freq =
            torch::from_blob(vec.data(), vec.size(), torch::TensorOptions().dtype(torch::kDouble))
                    .to(options)
                    .reciprocal();
    return inv_freq;
}

at::Tensor RotaryEmbeddingImpl::forward(at::Tensor &qkv) {
    // Input is NT3HD
    assert_forward_dims(qkv);
    const int64_t N = qkv.size(0);
    const int64_t T = qkv.size(1);
    const int64_t H = qkv.size(3);
    const int64_t D = qkv.size(4);

    auto buffers = named_buffers();
    const at::Tensor cos_buf = buffers["cos_freqs"].narrow(0, 0, T);
    const at::Tensor sin_buf = buffers["sin_freqs"].narrow(0, 0, T);

    auto qk_evens = qkv.slice(2, 0, 2).slice(4, 0, D / 2);
    auto qk_odds = qkv.slice(2, 0, 2).slice(4, D / 2, D);

    // Allocate output tensor with memory layout as consumed by attention: 3NHTD
    auto output = at::empty({3, N, H, T, D}, qkv.options());
    // View as [3 (q|k|v), N, H, T, 2 (even|odd), D/2], narrow first dim to 2 (q|k), then
    // permute as [2 (even|odd), N, T, 2 (q|k), H, D/2] which is compatible with assignment below
    auto output_kv_even_odd =
            output.view({3, N, H, T, 2, D / 2}).slice(0, 0, 2).permute({4, 1, 3, 0, 2, 5});

    // Apply rotary embedding to Q and K
    output_kv_even_odd[0] = cos_buf * qk_evens - sin_buf * qk_odds;
    output_kv_even_odd[1] = sin_buf * qk_evens + cos_buf * qk_odds;

    // Copy V to output
    output.select(0, 2).permute({0, 2, 1, 3}) = qkv.select(2, 2);

    return output;
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
          // TODO: this may benefit from fine-tuning. 8 gives good performance at chunk size 12k
          num_splits(utils::get_dev_opt<int>("mha_num_splits", 12)),
          attn_window(attn_window_),
          options(options_) {
    wqkv = register_module("wqkv", Linear(LinearOptions(d_model, 3 * d_model).bias(qkv_bias_)));
    out_proj = register_module("out_proj", Linear(LinearOptions(d_model, d_model).bias(out_bias_)));
    const float theta = 10000.0f;
    const int64_t max_seq_len = 2048;
    rotary_emb =
            register_module("rotary_emb", RotaryEmbedding(head_dim, theta, max_seq_len, options));
};

at::Tensor MultiHeadAttentionImpl::get_attn_window_mask(const int64_t size) {
    const auto key = MaskKey{size, options.device()};
    if (mask_cache.find(key) == mask_cache.end()) {
        mask_cache[key] = build_attn_window_mask(size);
    }
    return mask_cache.at(key);
}

at::Tensor MultiHeadAttentionImpl::build_attn_window_mask(const int64_t size) const {
    utils::ScopedProfileRange spr("AWM", 3);
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

#if DORADO_CUDA_BUILD
    bool use_koi_rote =
            x.is_cuda() && utils::get_dev_opt<bool>("use_koi_rote", true) && d_model <= 512;
    if (use_koi_rote) {
        if (!wqkv_transposed) {
            auto w = wqkv->weight;
            wqkv->weight.view({3, -1, head_dim / 2, 2, d_model}).slice(0, 0, 2) =
                    wqkv->weight.view({3, -1, 2, head_dim / 2, d_model})
                            .slice(0, 0, 2)
                            .transpose(2, 3)
                            .clone();
            if (wqkv->bias.numel()) {
                wqkv->bias.view({3, -1, head_dim / 2, 2}).slice(0, 0, 2) =
                        wqkv->bias.view({3, -1, 2, head_dim / 2})
                                .slice(0, 0, 2)
                                .transpose(2, 3)
                                .clone();
            }
            wqkv_transposed = true;
        }
    }
#endif
    at::Tensor qkv;
    at::Tensor attn_output_ntc;
    {
        utils::ScopedProfileRange spr("QKV", 3);
        // in_feat=512, out_feat=1536 (3*in), nhead=8, head_dim=64=(512/8), dim_ff=2048
        qkv = wqkv(x).view({N, T, 3, nhead, head_dim});
    }
    {
        utils::ScopedProfileRange spr("ROTE", 3);
#if DORADO_CUDA_BUILD
        if (use_koi_rote) {
            auto stream = at::cuda::getCurrentCUDAStream().stream();
            auto out = torch::empty({3, N, nhead, T, head_dim}, qkv.options());
            int res = host_rotary_embed_transpose_f16(
                    stream, static_cast<int>(N), static_cast<int>(T), nhead, head_dim,
                    rotary_emb->theta, qkv.data_ptr(), out.data_ptr());
            if (res != KOI_SUCCESS) {
                throw std::runtime_error("Koi rotary embedding failed.");
            }
            qkv = out;
        } else
#endif
        {
            qkv = rotary_emb(qkv);
        }
    }
    attn_output_ntc = at::empty({N, T, C}, x.options());
#if DORADO_CUDA_BUILD && !defined(DORADO_TX2)
    int res = KOI_NOT_SUPPORTED;
    bool use_koi_attention = x.is_cuda() && utils::get_dev_opt<bool>("use_koi_attention", true) &&
                             koi_can_use_cutlass();
    if (use_koi_attention) {
        utils::ScopedProfileRange spr("KOI_MEA", 3);
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        const auto [win_upper, win_lower] = attn_window;
        res = host_masked_attention_f16(stream, static_cast<int>(N), static_cast<int>(T), nhead,
                                        head_dim, win_upper, win_lower, qkv[0].data_ptr(),
                                        qkv[1].data_ptr(), qkv[2].data_ptr(),
                                        attn_output_ntc.data_ptr());
        if (res != KOI_SUCCESS && res != KOI_NOT_SUPPORTED) {
            throw std::runtime_error("Koi windowed attention failed.");
        }
    }
    if (res == KOI_NOT_SUPPORTED)
#endif
    {
        utils::ScopedProfileRange spr("MEA", 3);
        auto attn_window_mask = get_attn_window_mask(T);
        auto attn_output = attn_output_ntc.view({N, T, nhead, head_dim}).transpose(1, 2);
        const auto [win_upper, win_lower] = attn_window;
        // The MPS backend refuses to work on a span of the mask that doesn't have an
        // alignment of 4 elements, so pad the amount we process each loop to that.
        const auto elems_per_split =
                utils::pad_to(utils::div_round_up(T, int64_t{num_splits}), int64_t{4});
        for (int i = 0; i < num_splits; ++i) {
            const auto qb = i * elems_per_split;
            if (qb >= T) {
                break;
            }
            const auto qe = std::min(T, qb + elems_per_split);
            const auto kvb = std::max<int64_t>(0, qb - win_lower);
            const auto kve = std::min<int64_t>(T, qe + win_upper);
            const auto q = qkv[0].slice(-2, qb, qe);
            const auto k = qkv[1].slice(-2, kvb, kve);
            const auto v = qkv[2].slice(-2, kvb, kve);
            const auto mask = attn_window_mask.index({Slice(qb, qe), Slice(kvb, kve)});
#if TORCH_VERSION_MAJOR >= 2
            c10::optional<at::Tensor> opt_mask;
            // Not using the mask gets us significantly better performance, at the cost of some
            // accuracy. Accuracy loss is minimised by larger num_splits.
            if (utils::get_dev_opt<bool>("mha_use_mask", true)) {
                opt_mask = mask;
            }
            attn_output.slice(-2, qb, qe) = at::scaled_dot_product_attention(q, k, v, opt_mask);
#else
            attn_output.slice(-2, qb, qe) = scaled_dot_product_attention_naive(q, k, v, mask);
#endif
        }
    }
    {
        utils::ScopedProfileRange spr("OUTP", 3);
        x = out_proj(attn_output_ntc);
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
    const auto deepnorm_alpha = named_buffers()["deepnorm_alpha"];
#if DORADO_CUDA_BUILD
    const int N = static_cast<int>(x.size(0));
    const int T = static_cast<int>(x.size(1));
    const int C = static_cast<int>(x.size(2));
    x = x.contiguous();  // If using koi, make sure x is NTC order in memory
    const int num_rows = N * T;
#endif

    auto run_norm = [&](RMSNorm norm, const at::Tensor &in) {
#if DORADO_CUDA_BUILD
        int res = KOI_NOT_SUPPORTED;
        if (x.is_cuda()) {
            auto stream = at::cuda::getCurrentCUDAStream().stream();
            res = host_fused_residual_rmsnorm_f16(stream, C, num_rows, in.data_ptr(), x.data_ptr(),
                                                  deepnorm_alpha.data_ptr(),
                                                  norm->weight.data_ptr(), x.data_ptr());
            if (res != KOI_SUCCESS && res != KOI_NOT_SUPPORTED) {
                throw std::runtime_error("Koi error during layer norm");
            }
        }
        if (res == KOI_NOT_SUPPORTED)
#endif
        {
            x = norm(in + (x * deepnorm_alpha));
        }
    };

    {
        utils::ScopedProfileRange spr("MHE", 2);
        attn = self_attn(x);
    }
    {
        utils::ScopedProfileRange spr("LNORM1", 2);
        run_norm(norm1, attn);
    }
    {
        utils::ScopedProfileRange spr("FF", 2);
        f = ff(x);
    }
    {
        utils::ScopedProfileRange spr("LNORM2", 2);
        run_norm(norm2, f);
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

at::Tensor LinearScaledCRFImpl::forward(const at::Tensor &x) {
    if (!scale_applied) {
        linear->weight *= m_params.scale;
        scale_applied = true;
    }
    return linear(x);
}

TxModelImpl::TxModelImpl(const basecall::CRFModelConfig &config, const at::TensorOptions &options)
        : m_options(options) {
    convs = register_module("convs", basecall::nn::ConvStack(config.convs));
    tx_encoder = register_module("transformer_encoder", TxEncoderStack(config, m_options));
    tx_decoder = register_module("transformer_decoder", LinearUpsample(config.tx->upsample));
    crf = register_module("crf", LinearScaledCRF(config.tx->crf));
}

at::Tensor TxModelImpl::forward(const at::Tensor &chunk_NCT) {
    at::Tensor h;
    {
        utils::ScopedProfileRange spr("Conv", 1);
        // Returns: NTC
        h = convs->forward(chunk_NCT);
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
    // Returns: NTC
    return h;
}

}  // namespace nn

}  // namespace dorado::basecall
