#include "TxModel.h"

#include "basecall/nn/CRFModel.h"
#include "config/BasecallModelConfig.h"
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

namespace {
bool koi_can_use_cutlass() {
    cudaDeviceProp *prop = at::cuda::getCurrentDeviceProperties();
    return ((prop->major == 8 || prop->major == 9) && prop->minor == 0);
}

struct KoiTensorExt : public KoiTensor {
    KoiTensorExt(const at::Tensor &t, const std::vector<int> &dim_tags) { init(t, dim_tags); }
    KoiTensorExt(const at::Tensor &t,
                 const std::vector<int> &dim_tags,
                 const at::Tensor &scale_t,
                 int scale_tag_) {
        init(t, dim_tags);
        if (scale_t.numel()) {
            if (type_id != KOI_I8) {
                throw std::runtime_error("KoiTensor scale tensor only compatible with I8 tensor");
            }
            scale_tag = scale_tag_;
            scale_data = scale_t.data_ptr();
            scale_type_id = type_id_from_tensor(scale_t);
            if (scale_type_id != KOI_F32) {
                throw std::runtime_error("KoiTensor scale tensor must be F32");
            }
        }
    }

    KoiTypeId type_id_from_tensor(const at::Tensor &t) {
        if (t.dtype() == torch::kF16) {
            return KOI_F16;
        } else if (t.dtype() == torch::kF32) {
            return KOI_F32;
        } else if (t.dtype() == torch::kI8) {
            return KOI_I8;
        }
        throw std::runtime_error("KoiTensor unsupported dtype");
    }

    void init(const at::Tensor &t, const std::vector<int> &dim_tags) {
        data_ptr = t.data_ptr();
        ndims = static_cast<int>(t.dim());
        type_id = type_id_from_tensor(t);
        if (ndims > KOI_TENSOR_MAX_DIMS || size_t(ndims) != dim_tags.size()) {
            spdlog::error("KoiTensor dimension mismatch");
            exit(EXIT_FAILURE);
        }
        for (int i = 0; i < ndims; ++i) {
            dims[i].tag = dim_tags[i];
            dims[i].size = t.size(i);
            dims[i].stride = t.stride(i);
        }
        scale_data = nullptr;
        scale_tag = 0;
        scale_type_id = KOI_NONE;
    }
};
}  // anonymous namespace

#endif

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace dorado::basecall::nn {

namespace {

using namespace config;
using namespace torch::nn;
namespace Idx = torch::indexing;
using Slice = torch::indexing::Slice;

#if DORADO_CUDA_BUILD && !DORADO_TX2
void apply_rounding(at::Tensor &t, int remove_bits) {
    // Round Float16 tensor elements such that the last `remove_bits` of the mantissa are 0s.
    // TODO: this is slightly dangerous as it will turn numbers close to +/-65304 into +/-inf
    t.view(torch::kI16).add_(1 << (remove_bits - 1));
    t.view(torch::kI16).bitwise_and_(0x10000 - (1 << remove_bits));
}
#endif

}  // namespace

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
#if DORADO_CUDA_BUILD && !DORADO_TX2
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
                                               float theta_,
                                               int max_seq_len_,
                                               const at::TensorOptions &options_)
        : d_model(d_model_),
          nhead(nhead_),
          head_dim(d_model_ / nhead_),
          // TODO: this may benefit from fine-tuning. 8 gives good performance at chunk size 12k
          num_splits(utils::get_dev_opt<int>("mha_num_splits", 12)),
          attn_window(attn_window_),
          theta(theta_),
          max_seq_len(max_seq_len_),
          options(options_) {
    wqkv = register_module("wqkv", Linear(LinearOptions(d_model, 3 * d_model).bias(qkv_bias_)));
    out_proj = register_module("out_proj", Linear(LinearOptions(d_model, d_model).bias(out_bias_)));
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
#if DORADO_CUDA_BUILD && !DORADO_TX2
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

TxEncoderImpl::TxEncoderImpl(const TxEncoderParams &params_, const at::TensorOptions &options)
        : params(params_) {
    self_attn =
            register_module("self_attn", MultiHeadAttention(params.d_model, params.nhead, false,
                                                            true, params.attn_window, params.theta,
                                                            params.max_seq_len, options));
    ff = register_module("ff", GatedMLP(params.d_model, params.dim_feedforward));
    norm1 = register_module("norm1", RMSNorm(params.d_model));
    norm2 = register_module("norm2", RMSNorm(params.d_model));

    const at::Tensor deepnorm_alpha = at::tensor(params.deepnorm_alpha);
    register_buffer("deepnorm_alpha", deepnorm_alpha);
};

#if DORADO_CUDA_BUILD && !DORADO_TX2
void TxEncoderImpl::remove_bits() {
    // Round weights, zeroing the lowest mantissa bits (this makes the matmuls more
    // power efficient and results in higher performance for a small accuracy drop)
    int default_remove = utils::get_dev_opt("remove_bits", 4);
    apply_rounding(wqkv_weights_f16.t, utils::get_dev_opt("remove_bits_qkv", default_remove));
    apply_rounding(proj_weight, utils::get_dev_opt("remove_bits_proj", default_remove));
    apply_rounding(t_res_weights, utils::get_dev_opt("remove_bits_res", default_remove));
    apply_rounding(t_fc2_wts, utils::get_dev_opt("remove_bits_fc2", default_remove));
    apply_rounding(t_fc1_wts_f16.t, utils::get_dev_opt("remove_bits_fc1", default_remove));
    apply_rounding(t_res2_weights, utils::get_dev_opt("remove_bits_res2", default_remove));
}
#endif

void TxEncoderImpl::koi_forward(utils::ScaledTensor &scaled_tensor, at::Tensor &x_f16) {
    (void)scaled_tensor;
    (void)x_f16;
#if DORADO_CUDA_BUILD && !DORADO_TX2
    const int N = static_cast<int>(x_f16.size(0));
    const int T = static_cast<int>(x_f16.size(1)) * 16;
    const int C = params.d_model;
    const int D = params.d_model / params.nhead;
    const int H = params.nhead;
    const int E = params.dim_feedforward * 2;
    const auto [win_upper, win_lower] = params.attn_window;
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // Atomic counter, need 4 buffers of 4B each of working memory
    auto ctr = torch::zeros({4}, x_f16.options().dtype(torch::kI32));

    utils::ScopedProfileRange layer_spr("TxLayerKoiTiled", 2);
    const float alpha = params.deepnorm_alpha;

    auto f16_opts = x_f16.options().dtype(torch::kF16);

    if (!t_res_weights.numel()) {
        // Weights for the Q,K and V tensors which will be multiplied with the inputs
        wqkv_weights_f16.t = torch::empty_like(self_attn->wqkv->weight);
        // For Q and K weights, we need to take the lower and upper halves of the D dimension
        // and interleave them
        auto wqk = self_attn->wqkv->weight.view({3, H, 2, D / 2, C}).slice(0, 0, 2);
        wqkv_weights_f16.t.view({3, H, D / 2, 2, C}).slice(0, 0, 2) = wqk.transpose(2, 3);
        wqkv_weights_f16.t = wqkv_weights_f16.t.view({3, H, D, C});
        // Straight copy of V weights
        wqkv_weights_f16.t[2] = self_attn->wqkv->weight.view({3, H, D, C})[2];

        // Quantize, take rcp of scale
        wqkv_weights_i8 = utils::quantize_tensor(wqkv_weights_f16.t, -1);
        wqkv_weights_i8.scale.reciprocal_();

        // Rearrange qkv weights to tiled format
        wqkv_weights_i8.t = wqkv_weights_i8.t.view({3, H, D / 16, 16, C / 16, 16})
                                    .transpose(-3, -2)
                                    .contiguous();
        wqkv_weights_f16.t = wqkv_weights_f16.t.view({3, H, D / 16, 16, C / 8, 8})
                                     .transpose(-3, -2)
                                     .contiguous();

        //Rotary embedding as a torch tensor
        auto rot_bfrs = self_attn->rotary_emb->named_buffers();
        auto max_T = 16 * (rot_bfrs["sin_freqs"].size(0) / 16);
        sincos_bfr = torch::empty({max_T, D / 2, 2}, f16_opts);
        sincos_bfr.select(2, 0) = rot_bfrs["sin_freqs"].slice(0, 0, max_T).view({max_T, D / 2});
        sincos_bfr.select(2, 1) = rot_bfrs["cos_freqs"].slice(0, 0, max_T).view({max_T, D / 2});
        sincos_bfr = sincos_bfr.view({max_T / 16, 16, D / 8, 8}).transpose(1, 2).contiguous();

        // Need rearranging to correct tiled format
        proj_weight = self_attn->out_proj->weight.view({C / 16, 16, C / 8, 8})
                              .transpose(1, 2)
                              .contiguous();
        proj_bias = self_attn->out_proj->bias.view({C});

        t_res_weights = norm1->weight.view({C});
        t_res2_weights = norm2->weight.view({C});

        // Quantize SwiGLU weights, interleave, and rearrange as tiled
        auto fc1_weight_interleaved =
                ff->fc1->weight.unflatten(0, {2, -1, 16}).transpose(0, 1).contiguous();
        t_fc1_wts_i8 = utils::quantize_tensor(fc1_weight_interleaved, -1);
        t_fc1_wts_i8.scale.reciprocal_();
        t_fc1_wts_i8.t = t_fc1_wts_i8.t.view({E / 16, 16, C / 16, 16}).transpose(1, 2).contiguous();

        t_fc1_wts_f16.t =
                fc1_weight_interleaved.view({E / 16, 16, C / 8, 8}).transpose(1, 2).contiguous();

        t_fc2_wts = ff->fc2->weight.view({C / 16, 16, E / 16, 8}).transpose(1, 2).contiguous();

        this->remove_bits();
    }

    // Output buffers
    auto qkv = torch::empty({N, T / 16, 3, H, D / 8, 16, 8}, f16_opts);
    auto t_out_attn = torch::empty({N, T / 16, H, D / 8, 16, 8}, f16_opts);
    auto t_out_proj = torch::empty({N, T / 16, C / 8, 16, 8}, f16_opts);
    auto t_fc1_out = torch::empty({N * T / 16, E / 16, 16, 8}, f16_opts);
    auto t_fc2_out = torch::empty({N, T / 16, C / 8, 16, 8}, f16_opts);

    //Define Koi Tensors for the above buffers.
    auto &x = scaled_tensor.t;
    bool use_i8 = x.dtype() == torch::kI8;
    auto &wqkv_weights = use_i8 ? wqkv_weights_i8 : wqkv_weights_f16;
    auto &t_fc1_wts = use_i8 ? t_fc1_wts_i8 : t_fc1_wts_f16;
    KoiTensorExt in_f16(x_f16, {'N', 'T', 'C', 't', 'c'});
    KoiTensorExt in(x, {'N', 'T', 'C', 't', 'c'}, scaled_tensor.scale, 'C');
    KoiTensorExt *in_i8_ptr = use_i8 ? &in : nullptr;
    KoiTensorExt in_mk(x.flatten(0, 1), {'M', 'K', 'm', 'k'}, scaled_tensor.scale, 'K');
    KoiTensorExt weights_qkv(wqkv_weights.t, {KOI_DIM_MAT_Q_K_V, 'H', 'D', 'C', 'd', 'c'},
                             wqkv_weights.scale, 'C');
    KoiTensorExt out_qkv(qkv, {'N', 'T', KOI_DIM_MAT_Q_K_V, 'H', 'D', 't', 'd'});
    KoiTensorExt out_attn(t_out_attn, {'N', 'T', 'H', 'D', 't', 'd'});
    KoiTensorExt out_attn_mk(t_out_attn.flatten(2, 3).flatten(0, 1), {'M', 'K', 'm', 'k'});
    KoiTensorExt out_proj_mn(t_out_proj.flatten(0, 1), {'M', 'N', 'm', 'n'});
    KoiTensorExt out_proj_ntc(t_out_proj, {'N', 'T', 'C', 't', 'c'});
    KoiTensorExt fc1_wts(t_fc1_wts.t, {'N', 'K', 'n', 'k'}, t_fc1_wts.scale, 'K');
    KoiTensorExt fc1_out(t_fc1_out, {'M', 'N', 'm', 'n'});
    KoiTensorExt fc1_out_mk(t_fc1_out, {'M', 'K', 'm', 'k'});
    KoiTensorExt fc2_out_mn(t_fc2_out.flatten(0, 1), {'M', 'N', 'm', 'n'});
    KoiTensorExt fc2_out_ntc(t_fc2_out, {'N', 'T', 'C', 't', 'c'});

    KoiTensorExt sincos(sincos_bfr.slice(0, 0, T / 16), {'T', 'D', 't', 'd'});
    KoiTensorExt proj_w(proj_weight, {'N', 'K', 'n', 'k'});
    KoiTensorExt proj_b(proj_bias, {'N'});
    KoiTensorExt res_weights(t_res_weights, {'C'});
    KoiTensorExt fc2_wts(t_fc2_wts, {'N', 'K', 'n', 'k'});
    KoiTensorExt res2_weights(t_res2_weights, {'C'});

    int res = KOI_SUCCESS;
    int calls = 0;
    if (res == KOI_SUCCESS && ++calls) {
        // Fused QKV Matmul Plus Rotary embedding
        utils::ScopedProfileRange spr("QKV+ROTE", 3);
        res = koi_qkv_rotary(stream, self_attn->rotary_emb->theta, &in, &weights_qkv, &sincos,
                             &out_qkv, ctr[0].data_ptr<int>());
    }
    if (res == KOI_SUCCESS && ++calls) {
        // Apply masket attention
        utils::ScopedProfileRange spr("MEA", 3);
        res = koi_masked_attention(stream, win_upper, win_lower, &out_qkv, &out_attn);
    }
    if (res == KOI_SUCCESS && ++calls) {
        // Koi linear matmul
        utils::ScopedProfileRange spr("OUTP", 3);
        res = koi_linear(stream, &out_attn_mk, &proj_w, &proj_b, &out_proj_mn,
                         ctr[1].data_ptr<int>());
    }
    if (res == KOI_SUCCESS && ++calls) {
        // RMS residual
        utils::ScopedProfileRange spr("LNORM1", 3);
        res = koi_rmsnorm_residual(stream, &out_proj_ntc, &in_f16, alpha, &res_weights, &in_f16,
                                   in_i8_ptr);
    }
    if (res == KOI_SUCCESS && ++calls) {
        // Matmul + SWIGLU
        utils::ScopedProfileRange spr("FC1+SILU", 3);
        int use_f32_accum = int(utils::get_dev_opt<bool>("koi_swiglu_f32_accum", false));
        res = koi_mm_swiglu(stream, &in_mk, &fc1_wts, &fc1_out, ctr[2].data_ptr<int>(),
                            use_f32_accum);
    }
    if (res == KOI_SUCCESS && ++calls) {
        // Fully connected
        utils::ScopedProfileRange spr("FC2", 3);
        res = koi_linear(stream, &fc1_out_mk, &fc2_wts, nullptr, &fc2_out_mn,
                         ctr[3].data_ptr<int>());
    }
    if (res == KOI_SUCCESS && ++calls) {
        // RMS Norm Residual again
        utils::ScopedProfileRange spr("LNORM2", 3);
        res = koi_rmsnorm_residual(stream, &fc2_out_ntc, &in_f16, alpha, &res2_weights, &in_f16,
                                   in_i8_ptr);
    }
    // TODO: handle result
    if (res != KOI_SUCCESS) {
        spdlog::error("Koi tiled path failed {}", calls);
    }
#endif
}

void TxEncoderImpl::koi_volta_forward(at::Tensor &x_f16) {
    (void)x_f16;
#if DORADO_CUDA_BUILD && !DORADO_TX2 && !DORADO_ORIN
    const int N = static_cast<int>(x_f16.size(0));
    const int T = static_cast<int>(x_f16.size(1)) * 16;
    const int C = params.d_model;
    const int D = params.d_model / params.nhead;
    const int H = params.nhead;
    const int E = params.dim_feedforward * 2;
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // Koi tests show fastest configuration with f32_accum for proj, swiglu, fc2 and f16_accum for qkv
    bool useFloatAccumQKV = utils::get_dev_opt("volta_f32_accum_qkv", false);
    bool useFloatAccumProj = utils::get_dev_opt("volta_f32_accum_proj", false);
    bool useBiasProj = true;
    bool useFloatAccumSwiglu = utils::get_dev_opt("volta_f32_accum_swiglu", false);
    bool useFloatAccumfc2 = utils::get_dev_opt("volta_f32_accum_fc2", false);
    bool useBiasfc2 = false;  // No bias for fc2

    utils::ScopedProfileRange layer_spr("TxLayerKoiVoltaTiled", 2);
    const float alpha = params.deepnorm_alpha;

    auto f16_opts = x_f16.options().dtype(torch::kF16);
    if (!t_res_weights.numel()) {
        // Weights for the Q,K and V tensors which will be multiplied with the inputs
        wqkv_weights_f16.t = torch::empty_like(self_attn->wqkv->weight);
        // For Q and K weights, we need to take the lower and upper halves of the D dimension
        // and interleave them
        auto wqk = self_attn->wqkv->weight.view({3, H, 2, D / 2, C}).slice(0, 0, 2);
        wqkv_weights_f16.t.view({3, H, D / 2, 2, C}).slice(0, 0, 2) = wqk.transpose(2, 3);
        wqkv_weights_f16.t = wqkv_weights_f16.t.view({3, H, D, C});
        // Straight copy of V weights
        wqkv_weights_f16.t[2] = self_attn->wqkv->weight.view({3, H, D, C})[2];

        wqkv_weights_f16.t = wqkv_weights_f16.t.view({(3 * H * D) / 16, 16, C / 16, 16})
                                     .permute({0, 2, 1, 3})
                                     .contiguous();

        //Rotary embedding as a torch tensor
        auto rot_bfrs = self_attn->rotary_emb->named_buffers();
        auto max_T = 16 * (rot_bfrs["sin_freqs"].size(0) / 16);
        sincos_bfr = torch::empty({max_T, D / 2, 2}, f16_opts);
        sincos_bfr.select(2, 0) = rot_bfrs["sin_freqs"].slice(0, 0, max_T).view({max_T, D / 2});
        sincos_bfr.select(2, 1) = rot_bfrs["cos_freqs"].slice(0, 0, max_T).view({max_T, D / 2});
        sincos_bfr = sincos_bfr.view({max_T / 16, 2, 2, 4, D / 16, 2, 8})
                             .permute({0, 4, 5, 2, 1, 3, 6})
                             .contiguous();  // This follows volta mma layout 16x16 tile

        // proj_weight is the M, N=512, K=512 linear layer. self_attn->out_proj->weight is of layout (N, K)
        proj_weight = self_attn->out_proj->weight.view({C / 16, 16, C / 16, 16})
                              .transpose(1, 2)
                              .contiguous();  // NKnk
        proj_bias = self_attn->out_proj->bias.view({C});

        // Both RMSNorm weights
        t_res_weights = norm1->weight.view({C});
        t_res2_weights = norm2->weight.view({C});

        // Quantize SwiGLU weights, interleave, and rearrange as tiled
        auto fc1_weight_interleaved =
                ff->fc1->weight.unflatten(0, {2, -1, 16}).transpose(0, 1).contiguous();

        t_fc1_wts_f16.t =
                fc1_weight_interleaved.view({E / 16, 16, C / 16, 16}).transpose(1, 2).contiguous();

        // t_fc2_wts is the M, N=512=C, K=2048 linear layer. ff->fc2->weight is of shape (N, K)
        t_fc2_wts =
                ff->fc2->weight.view({C / 16, 16, (E / 2) / 16, 16}).transpose(1, 2).contiguous();

        this->remove_bits();
    }

    // Output buffers
    auto qkv = torch::empty({N, T / 16, 3, H, D / 16, 16, 16}, f16_opts);
    auto t_out_attn = torch::empty({N, T / 16, H, D / 16, 16, 16}, f16_opts);
    auto t_out_proj = torch::empty({N, T / 16, C / 16, 16, 16}, f16_opts);
    auto t_out_rms1 = torch::empty({N, T / 16, C / 16, 16, 16}, f16_opts);
    auto t_fc1_out = torch::empty({N * T / 16, (E / 2) / 16, 16, 16}, f16_opts);
    auto t_fc2_out = torch::empty({N, T / 16, C / 16, 16, 16}, f16_opts);

    int res = KOI_SUCCESS;
    int calls = 0;
    std::stringstream ss;
    if (res == KOI_SUCCESS && ++calls) {
        // Fused QKV Matmul Plus Rotary embedding
        utils::ScopedProfileRange spr("QKV+ROTE Volta", 3);
        res = koi_volta_qkv_rotary(stream, x_f16.data_ptr(), wqkv_weights_f16.t.data_ptr(),
                                   qkv.data_ptr(), sincos_bfr.data_ptr(), useFloatAccumQKV, N, T);
    }
    if (res == KOI_SUCCESS && ++calls) {
        // Apply window flashattention
        utils::ScopedProfileRange spr("MEA Volta", 3);
        res = koi_volta_attn(stream, qkv.data_ptr(), t_out_attn.data_ptr(), N, T);
    }
    if (res == KOI_SUCCESS && ++calls) {
        // Koi linear matmul, proj weights (K=512)
        utils::ScopedProfileRange spr("OUTP Volta", 3);
        res = koi_volta_linear(stream, t_out_attn.data_ptr(), proj_weight.data_ptr(),
                               t_out_proj.data_ptr(), proj_bias.data_ptr(), useFloatAccumProj,
                               useBiasProj, N * T, C);
    }
    if (res == KOI_SUCCESS && ++calls) {
        // RMS residual
        utils::ScopedProfileRange spr("LNORM1 Volta", 3);
        res = koi_volta_rmsnorm_residual(stream, x_f16.data_ptr(), t_out_proj.data_ptr(),
                                         t_res_weights.data_ptr(), t_out_rms1.data_ptr(), alpha,
                                         N * T);
    }
    if (res == KOI_SUCCESS && ++calls) {
        // Matmul + SWIGLU
        utils::ScopedProfileRange spr("FC1+SILU Volta", 3);
        koi_volta_mm_swiglu(stream, t_out_rms1.data_ptr(), t_fc1_wts_f16.t.data_ptr(),
                            t_fc1_out.data_ptr(), useFloatAccumSwiglu, N * T);
    }
    if (res == KOI_SUCCESS && ++calls) {
        // Koi linear matmul, fc2 weights (K=2048)
        utils::ScopedProfileRange spr("FC2 Volta", 3);
        res = koi_volta_linear(stream, t_fc1_out.data_ptr(), t_fc2_wts.data_ptr(),
                               t_fc2_out.data_ptr(),
                               nullptr,  // No bias for fc2
                               useFloatAccumfc2, useBiasfc2, N * T, E / 2);
    }
    if (res == KOI_SUCCESS && ++calls) {
        // RMS Norm Residual again
        utils::ScopedProfileRange spr("LNORM2 Volta", 3);
        res = koi_volta_rmsnorm_residual(
                stream, t_out_rms1.data_ptr(), t_fc2_out.data_ptr(), t_res2_weights.data_ptr(),
                // Use initial input x_f16 as output buffer for second rmsnorm
                x_f16.data_ptr(), alpha, N * T);
    }
    if (res != KOI_SUCCESS) {
        spdlog::error("Koi Volta tiled path failed {}", calls);
    }
#endif
}

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

TxEncoderStackImpl::TxEncoderStackImpl(const TxEncoderParams &params,
                                       const at::TensorOptions &options) {
#if DORADO_CUDA_BUILD && !DORADO_TX2
    // TODO: make sure these are all the requirements
    bool is_sup_model = (params.d_model == 512) && (params.nhead == 8) &&
                        (params.attn_window.first == 127) && (params.attn_window.second == 128) &&
                        (params.dim_feedforward == 2048);
    use_koi_tiled = is_sup_model && options.device().is_cuda() &&
                    koi_tc_is_available(KOI_F16) == KOI_SUCCESS;
    spdlog::debug("TxEncoderStack: use_koi_tiled {}.", use_koi_tiled);

#if !DORADO_ORIN
    // Custom Volta flag
    use_koi_volta_tiled = is_sup_model && options.device().is_cuda() &&
                          koi_volta_tc_is_available(KOI_F16) == KOI_SUCCESS;
    spdlog::debug("TxEncoderStack: use_koi_volta_tiled {}.", use_koi_volta_tiled);
#endif
#endif

    stack = Sequential();
    for (int i = 0; i < params.depth; ++i) {
        TxEncoder encoder(params, options);
        stack->push_back(register_module("transformer_encoder" + std::to_string(i), encoder));
        layer_vec.push_back(encoder);
    }
    use_i8 = utils::get_dev_opt<bool>("koi_use_i8", true);
};

at::Tensor TxEncoderStackImpl::forward(const at::Tensor &x) {
#if DORADO_CUDA_BUILD && !DORADO_TX2
    if (use_koi_tiled) {
        const int N = static_cast<int>(x.size(0));
        const int T = static_cast<int>(x.size(1));
        const int C = static_cast<int>(x.size(2));

        at::Tensor tiled_f16;
        {
            utils::ScopedProfileRange spr("Tile F16", 2);
            tiled_f16 = x.view({N, T / 16, 16, C / 8, 8}).transpose(2, 3).contiguous();
        }

        utils::ScaledTensor scaled_tensor;
        if (use_i8) {
            utils::ScopedProfileRange spr("Quantise/tile I8", 2);
            // Quanitze tensor to i8 and take the reciprocal of the scale.
            scaled_tensor = utils::quantize_tensor(x, 2);
            scaled_tensor.scale.reciprocal_();
            // Transform the tensor into the tiled format.
            scaled_tensor.t =
                    scaled_tensor.t.view({N, T / 16, 16, C / 16, 16}).transpose(2, 3).contiguous();
        } else {
            scaled_tensor.t = tiled_f16;
        }

        for (auto &layer : layer_vec) {
            layer->koi_forward(scaled_tensor, tiled_f16);
        }

        at::Tensor untiled_f16;
        {
            utils::ScopedProfileRange spr("Untile F16", 2);
            untiled_f16 = tiled_f16.transpose(2, 3).contiguous().view({N, T, C});
        }
        return untiled_f16;
    }
#if !DORADO_ORIN
    else if (use_koi_volta_tiled) {
        const int N = static_cast<int>(x.size(0));
        const int T = static_cast<int>(x.size(1));
        const int C = static_cast<int>(x.size(2));

        at::Tensor tiled_f16;
        {
            utils::ScopedProfileRange spr("Tile F16 Volta", 2);
            tiled_f16 = x.view({N, T / 16, 16, C / 16, 16}).transpose(2, 3).contiguous();
        }

        for (auto &layer : layer_vec) {
            layer->koi_volta_forward(tiled_f16);
        }

        at::Tensor untiled_f16;
        {
            utils::ScopedProfileRange spr("Untile F16 Volta", 2);
            untiled_f16 = tiled_f16.transpose(2, 3).contiguous().view({N, T, C});
        }
        return untiled_f16;
    }
#endif
#endif
    return stack->forward(x);
}

LinearUpsampleImpl::LinearUpsampleImpl(const EncoderUpsampleParams &params)
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

LinearScaledCRFImpl::LinearScaledCRFImpl(const CRFEncoderParams &params) {
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

TxModelImpl::TxModelImpl(const BasecallModelConfig &config, const at::TensorOptions &options)
        : m_options(options) {
    convs = register_module("convs", basecall::nn::ConvStack(config.convs));
    tx_encoder = register_module("transformer_encoder", TxEncoderStack(config.tx->tx, m_options));
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

}  // namespace dorado::basecall::nn
