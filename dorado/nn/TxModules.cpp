#include "nn/TxModules.h"

#include "config/BasecallModelConfig.h"
#include "nn/TxModules.h"
#include "torch_utils/gpu_profiling.h"
#include "utils/dev_utils.h"
#include "utils/log_utils.h"
#include "utils/math_utils.h"

#include <ATen/TensorIndexing.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <c10/core/ScalarType.h>

#include <cmath>

#if DORADO_CUDA_BUILD
#include "nn/KoiUtils.h"

extern "C" {
#include "koi.h"
}

namespace {

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
        } else if (t.dtype() == torch::kFloat8_e4m3fn) {
            return KOI_E4M3;
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
            DORADO_LOG(error, "KoiTensor dimension mismatch");
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

#include <stdexcept>
#include <string>
#include <vector>

namespace dorado::nn {

namespace {

using namespace config;
using namespace torch::nn;
using Slice = torch::indexing::Slice;

#if DORADO_CUDA_BUILD
void apply_rounding(at::Tensor &t, int remove_bits) {
    if (t.dtype() == torch::kF16) {
        // Round Float16 tensor elements such that the last `remove_bits` of the mantissa are 0s.
        // TODO: this is slightly dangerous as it will turn numbers close to +/-65304 into +/-inf
        t.view(torch::kI16).add_(1 << (remove_bits - 1));
        t.view(torch::kI16).bitwise_and_(0x10000 - (1 << remove_bits));
    }
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

GatedMLPImpl::GatedMLPImpl(int in_features_, int hidden_features_, bool use_uswiglu_)
        : in_features(in_features_), hidden_features(hidden_features_), use_uswiglu(use_uswiglu_) {
    fc1 = register_module("fc1",
                          Linear(LinearOptions(in_features, 2 * hidden_features).bias(false)));
    fc2 = register_module("fc2", Linear(LinearOptions(hidden_features, in_features).bias(false)));
};

at::Tensor GatedMLPImpl::forward(const at::Tensor &x) {
    at::Tensor t;
    {
        utils::ScopedProfileRange spr("FC1", 3);
        t = fc1(x);
    }
    {
        utils::ScopedProfileRange spr("SILU", 3);
        const auto chunks = t.chunk(2, -1);
        if (use_uswiglu) {
            // 1.702 and limit=7.0 are hard-coded for now
            const auto &y = chunks[0].clamp_(-7.0, 7.0);
            const auto &gate = chunks[1].clamp_max_(7.0);
            t = (gate * torch::sigmoid(1.702 * gate)).mul_(y);
        } else {
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
    at::Tensor inv_freq = get_inv_freqs();

    at::Tensor cos_freqs, sin_freqs;
    // The older method (`else` path) works fine for max_seq_len <= 2048 (i.e. previous maximum),
    // so in that case we stick with what's tried and tested. For larger chunksizes, however,
    // the old method can produce large errors. Override via `--devopts rope_init_iter=[0|1]`
    bool rope_init_iter = utils::get_dev_opt<bool>("rope_init_iter", max_seq_len > 2048);
    if (rope_init_iter) {
        // Initialise sin/cos buffers by iteratively rotating. This is much more numerically
        // stable than computing sin/cos from the frequencies directly (as in the else path below).
        // We might accumulate some drift in the angles over the sequence length, but this doesn't
        // matter since only the relative angles between Q and K positions matter, and windowed
        // attention limits the distance between the two.
        auto opts_f64 = torch::TensorOptions().dtype(torch::kDouble);
        cos_freqs = torch::empty({max_seq_len, 1, 1, dim / 2}, opts_f64);
        sin_freqs = torch::empty({max_seq_len, 1, 1, dim / 2}, opts_f64);
        cos_freqs[0] = 1;
        sin_freqs[0] = 0;
        inv_freq = inv_freq.to(cos_freqs.device());
        cos_freqs[1] = torch::cos(inv_freq);
        sin_freqs[1] = torch::sin(inv_freq);
        int renorm_step = utils::get_dev_opt<int>("rope_init_renorm_step", 16);
        for (int i = 2; i < max_seq_len; ++i) {
            cos_freqs[i] = cos_freqs[i - 1] * cos_freqs[1] - sin_freqs[i - 1] * sin_freqs[1];
            sin_freqs[i] = sin_freqs[i - 1] * cos_freqs[1] + cos_freqs[i - 1] * sin_freqs[1];
            if (i % renorm_step == 0) {
                auto norm = (cos_freqs[i] * cos_freqs[i] + sin_freqs[i] * sin_freqs[i]).rsqrt();
                cos_freqs[i].mul_(norm);
                sin_freqs[i].mul_(norm);
            }
        }
    } else {
        // freqs.shape := {max_seq_len, 1, 1, dim/2}
        const at::Tensor freqs =
                torch::arange(max_seq_len, options).reshape({max_seq_len, 1, 1, 1}) * inv_freq;
        cos_freqs = torch::cos(freqs);
        sin_freqs = torch::sin(freqs);
    }

    register_buffer("cos_freqs", cos_freqs.to(options));
    register_buffer("sin_freqs", sin_freqs.to(options));
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
    double rcp_dim = 1.0 / static_cast<double>(dim);
    for (int i = 0; i < dim; i += 2) {
        vec.push_back(std::pow(static_cast<double>(theta), static_cast<double>(i) * rcp_dim));
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
        DORADO_LOG(error,
                   "RotE - maximum sequence length exceeded (len:{} > max:{}) - "
                   "Your chunksize may be too large",
                   seq_len, max_seq_len);
    }
    if (three != 3) {
        has_error = true;
        DORADO_LOG(error, "RotE - expected constant size:3 at dim:2 found:{}", three);
    }
    if (head_dim != dim) {
        has_error = true;
        DORADO_LOG(error, "RotE - expected head_dim size:{} at dim:4 found:{}", dim, head_dim);
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
            c10::optional<at::Tensor> opt_mask;
            // Not using the mask gets us significantly better performance, at the cost of some
            // accuracy. Accuracy loss is minimised by larger num_splits.
            if (utils::get_dev_opt<bool>("mha_use_mask", true)) {
                opt_mask = mask;
            }
            attn_output.slice(-2, qb, qe) = at::scaled_dot_product_attention(q, k, v, opt_mask);
        }
    }
    {
        utils::ScopedProfileRange spr("OUTP", 3);
        x = out_proj(attn_output_ntc);
    }
    return x;
};

TxEncoderImpl::TxEncoderImpl(const TransformerEncoderLayerParams &params_,
                             const at::TensorOptions &options,
                             bool use_uswiglu_)
        : params(params_), use_uswiglu(use_uswiglu_) {
    self_attn =
            register_module("self_attn", MultiHeadAttention(params.d_model, params.nhead, false,
                                                            true, params.attn_window, params.theta,
                                                            params.max_seq_len, options));
    ff = register_module("ff", GatedMLP(params.d_model, params.dim_feedforward, use_uswiglu));
    norm1 = register_module("norm1", RMSNorm(params.d_model));
    norm2 = register_module("norm2", RMSNorm(params.d_model));

    const at::Tensor deepnorm_alpha = at::tensor(params.deepnorm_alpha);
    register_buffer("deepnorm_alpha", deepnorm_alpha);
};

#if DORADO_CUDA_BUILD
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

void TxEncoderImpl::koi_forward(utils::ScaledTensor &scaled_tensor,
                                at::Tensor &x_f16,
                                [[maybe_unused]] AuxiliaryData *const aux) {
    (void)scaled_tensor;
    (void)x_f16;
#if DORADO_CUDA_BUILD
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
    auto f8_opts = x_f16.options().dtype(torch::kFloat8_e4m3fn);
    bool use_f8 = (koi_tc_is_available(KOI_E4M3) == KOI_SUCCESS) &&
                  utils::get_dev_opt<bool>("koi_use_f8", true);

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
        if (aux && (max_T < aux->max_chunk_size_tx_enc())) {
            throw std::runtime_error(
                    "seq_length of pre-computed sincos RoPE buffers is not big enough to "
                    "accommodate requested chunk_size");
        }
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
        if (use_f8) {
            const auto fc1_weight_shuffle_index =
                    torch::tensor({0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15},
                                  x_f16.options().dtype(torch::kI32));
            fc1_weight_interleaved =
                    torch::index_select(fc1_weight_interleaved, 2, fc1_weight_shuffle_index)
                            .contiguous();

            t_fc1_wts_f8.t = fc1_weight_interleaved.to(torch::kFloat8_e4m3fn)
                                     .view({E / 16, 16, C / 16, 16})
                                     .transpose(1, 2)
                                     .contiguous();
            t_fc2_wts = ff->fc2->weight.to(torch::kFloat8_e4m3fn)
                                .view({C / 16, 16, E / 32, 16})
                                .transpose(1, 2)
                                .contiguous();
        } else {
            t_fc1_wts_i8 = utils::quantize_tensor(fc1_weight_interleaved, -1);
            t_fc1_wts_i8.scale.reciprocal_();
            t_fc1_wts_i8.t =
                    t_fc1_wts_i8.t.view({E / 16, 16, C / 16, 16}).transpose(1, 2).contiguous();
            t_fc2_wts = ff->fc2->weight.view({C / 16, 16, E / 16, 8}).transpose(1, 2).contiguous();
        }
        t_fc1_wts_f16.t =
                fc1_weight_interleaved.view({E / 16, 16, C / 8, 8}).transpose(1, 2).contiguous();

        this->remove_bits();
    }

    // Output buffers
    auto qkv = torch::empty({N, T / 16, 3, H, D / 8, 16, 8}, f16_opts);
    auto t_out_attn = torch::empty({N, T / 16, H, D / 8, 16, 8}, f16_opts);
    auto t_out_proj = torch::empty({N, T / 16, C / 8, 16, 8}, f16_opts);
    auto t_rms_out = torch::empty({N, T / 16, C / 16, 16, 16}, f8_opts);
    auto t_fc1_out = use_f8 ? torch::empty({N * T / 16, E / 32, 16, 16}, f8_opts)
                            : torch::empty({N * T / 16, E / 16, 16, 8}, f16_opts);
    auto t_fc2_out = torch::empty({N, T / 16, C / 8, 16, 8}, f16_opts);

    //Define Koi Tensors for the above buffers.
    auto &x = scaled_tensor.t;
    const bool use_i8 = (x.dtype() == torch::kI8);
    auto &wqkv_weights = use_i8 ? wqkv_weights_i8 : wqkv_weights_f16;
    auto &t_fc1_wts = use_f8 ? t_fc1_wts_f8 : (use_i8 ? t_fc1_wts_i8 : t_fc1_wts_f16);
    KoiTensorExt in_f16(x_f16, {'N', 'T', 'C', 't', 'c'});
    KoiTensorExt in(x, {'N', 'T', 'C', 't', 'c'}, scaled_tensor.scale, 'C');
    KoiTensorExt *in_i8_ptr = use_i8 ? &in : nullptr;
    KoiTensorExt in_mk =
            use_f8 ? KoiTensorExt{t_rms_out.flatten(0, 1), {'M', 'K', 'm', 'k'}}
                   : KoiTensorExt{x.flatten(0, 1), {'M', 'K', 'm', 'k'}, scaled_tensor.scale, 'K'};
    KoiTensorExt weights_qkv(wqkv_weights.t, {KOI_DIM_MAT_Q_K_V, 'H', 'D', 'C', 'd', 'c'},
                             wqkv_weights.scale, 'C');
    KoiTensorExt out_qkv(qkv, {'N', 'T', KOI_DIM_MAT_Q_K_V, 'H', 'D', 't', 'd'});
    KoiTensorExt out_attn(t_out_attn, {'N', 'T', 'H', 'D', 't', 'd'});
    KoiTensorExt out_attn_mk(t_out_attn.flatten(2, 3).flatten(0, 1), {'M', 'K', 'm', 'k'});
    KoiTensorExt out_proj_mn(t_out_proj.flatten(0, 1), {'M', 'N', 'm', 'n'});
    KoiTensorExt out_proj_ntc(t_out_proj, {'N', 'T', 'C', 't', 'c'});
    KoiTensorExt out_rms_f8(t_rms_out, {{'N', 'T', 'C', 't', 'c'}});
    KoiTensorExt fc1_out(t_fc1_out, {'M', 'N', 'm', 'n'});
    KoiTensorExt fc1_out_mk(t_fc1_out, {'M', 'K', 'm', 'k'});
    KoiTensorExt fc2_out_mn(t_fc2_out.flatten(0, 1), {'M', 'N', 'm', 'n'});
    KoiTensorExt fc2_out_ntc(t_fc2_out, {'N', 'T', 'C', 't', 'c'});

    KoiTensorExt sincos(sincos_bfr.slice(0, 0, (aux ? aux->max_chunk_size_tx_enc() : T) / 16),
                        {'T', 'D', 't', 'd'});
    KoiTensorExt proj_w(proj_weight, {'N', 'K', 'n', 'k'});
    KoiTensorExt proj_b(proj_bias, {'N'});

    int res = KOI_SUCCESS;
    int calls = 0;
    if (res == KOI_SUCCESS && ++calls) {
        // Fused QKV Matmul Plus Rotary embedding
        utils::ScopedProfileRange spr("QKV+ROTE", 3);
        res = koi_qkv_rotary(stream, &in, &weights_qkv, &sincos, &out_qkv,
                             aux ? aux->qkv_rope_lut.data_ptr<int>() : nullptr,
                             ctr[0].data_ptr<int>());
    }
    if (res == KOI_SUCCESS && ++calls) {
        // Apply masked attention
        utils::ScopedProfileRange spr("MEA", 3);
        if (aux) {
            res = koi_vcs_attn(stream, qkv.data_ptr(), aux->device_chunk_table.data_ptr<int>(),
                               aux->total_num_varlen_chunks(), t_out_attn.data_ptr());
        } else {
            res = koi_masked_attention(stream, win_upper, win_lower, &out_qkv, &out_attn);
        }
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
        KoiTensorExt res_weights(t_res_weights, {'C'});
        res = koi_rmsnorm_residual(stream, &out_proj_ntc, &in_f16, alpha, &res_weights, &in_f16,
                                   use_f8 ? &out_rms_f8 : nullptr, use_f8 ? nullptr : in_i8_ptr);
    }
    if (res == KOI_SUCCESS && ++calls) {
        // Matmul + SWIGLU
        utils::ScopedProfileRange spr("FC1+SILU", 3);
        KoiTensorExt fc1_wts(t_fc1_wts.t, {'N', 'K', 'n', 'k'}, t_fc1_wts.scale, 'K');
        int use_f32_accum = int(utils::get_dev_opt<bool>("koi_swiglu_f32_accum", false));
        res = koi_mm_swiglu(stream, &in_mk, &fc1_wts, &fc1_out, ctr[2].data_ptr<int>(),
                            use_f32_accum, use_uswiglu);
    }
    if (res == KOI_SUCCESS && ++calls) {
        // Fully connected
        utils::ScopedProfileRange spr("FC2", 3);
        KoiTensorExt fc2_wts(t_fc2_wts, {'N', 'K', 'n', 'k'});
        res = koi_linear(stream, &fc1_out_mk, &fc2_wts, nullptr, &fc2_out_mn,
                         ctr[3].data_ptr<int>());
    }
    if (res == KOI_SUCCESS && ++calls) {
        // RMS Norm Residual again
        utils::ScopedProfileRange spr("LNORM2", 3);
        KoiTensorExt res2_weights(t_res2_weights, {'C'});
        res = koi_rmsnorm_residual(stream, &fc2_out_ntc, &in_f16, alpha, &res2_weights, &in_f16,
                                   nullptr, in_i8_ptr);
    }
    // TODO: handle result
    if (res != KOI_SUCCESS) {
        DORADO_LOG(error, "Koi tiled path failed {}", calls);
    }
#endif
}

void TxEncoderImpl::koi_nvfp4_forward(at::Tensor &x_tiled_nvfp4, const int N, const int T) {
#if DORADO_CUDA_BUILD && !DORADO_ORIN
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int C = params.d_model;
    const int D = params.d_model / params.nhead;
    const int H = params.nhead;
    const int E = params.dim_feedforward * 2;
    // TODO: I should have alpha be an argument to RMSNorm kernels...
    // const float alpha = params.deepnorm_alpha;
    const int use_uswiglu = utils::get_dev_opt("unconventional_swiglu", 0);

    // auto f32_opts = x_tiled_nvfp4.options().dtype(torch::kFloat);
    auto f16_opts = x_tiled_nvfp4.options().dtype(torch::kF16);
    auto f8_opts = x_tiled_nvfp4.options().dtype(torch::kFloat8_e4m3fn);
    auto i32_opts = x_tiled_nvfp4.options().dtype(torch::kInt);
    if (!t_res_weights.numel()) {
        // QKV weights layout is very messy for now, sorry
        const int64_t SMEM_M_QKV = 128;
        const int64_t SMEM_N_QKV = 192;
        const int64_t SMEM_N = 128;
        const int64_t SMEM_K = 128;
        auto [wqkv_scaled_fp16, wqkv_sf_fp8] =
                to_block_fp4(self_attn->wqkv->weight.view({3 * H * D, C / 16, 16}), 6.f);

        auto wqk_scaled = wqkv_scaled_fp16.slice(0, 0, 2 * H * D);
        auto wv_scaled = wqkv_scaled_fp16.slice(0, 2 * H * D, 3 * H * D);
        wqk_scaled = wqk_scaled.view({(2 * H * D) / 16, 4, 2, 2, C})
                             .transpose(1, 2)
                             .contiguous()
                             .reshape({2 * H * D, C});
        wqk_scaled = wqk_scaled.view({(2 * H * D) / 32, 2, 2, 8, C / 64, 2, 4, 8})
                             .permute({0, 4, 1, 3, 6, 2, 5, 7})
                             .contiguous()
                             .reshape({(2 * H * D) / 32, C / 64, -1});
        wv_scaled = wv_scaled.view({(H * D) / 32, 2, 2, 8, C / 64, 2, 4, 8})
                            .permute({0, 4, 1, 3, 6, 5, 2, 7})
                            .contiguous()
                            .reshape({(H * D) / 32, C / 64, -1});

        wqkv_scaled_fp16 = torch::cat({wqk_scaled, wv_scaled}, 0);
        wqkv_scaled_fp16 = wqkv_scaled_fp16.view({3, (H * D) / 32, C / 64, -1})
                                   .transpose(0, 1)
                                   .contiguous()
                                   .reshape({(3 * H * D) / 32, C / 64, -1});
        wqkv_scaled_fp16 = wqkv_scaled_fp16
                                   .view({(3 * H * D) / SMEM_N_QKV, SMEM_N_QKV / 32, C / SMEM_K,
                                          SMEM_K / 64, -1})
                                   .permute({0, 2, 1, 3, 4})
                                   .contiguous();

        at::Tensor wqkv_fp4 = torch::empty(
                {(3 * H * D) / SMEM_N_QKV, C / SMEM_K, (SMEM_N_QKV / 32) * (SMEM_K / 64), 256},
                i32_opts);
        koi_convert_fp16_to_fp4(stream, wqkv_fp4.data_ptr(), wqkv_scaled_fp16.data_ptr(),
                                3 * H * D * C);

        at::Tensor wqk_sf = wqkv_sf_fp8.slice(0, 0, 2 * H * D);
        at::Tensor wv_sf = wqkv_sf_fp8.slice(0, 2 * H * D, 3 * H * D);
        wqk_sf = wqk_sf.view({(2 * H * D) / 16, 4, 2, 2, C / 16})
                         .transpose(1, 2)
                         .contiguous()
                         .reshape({2 * H * D, C / 16});
        wqkv_sf_fp8 = at::cat({wqk_sf, wv_sf}, 0);

        wqkv_sf_fp8 = wqkv_sf_fp8.view({3, (H * D) / 32, 32, C / 16})
                              .transpose(0, 1)
                              .contiguous()
                              .reshape({3 * H * D, C / 16});
        std::vector<int64_t> shape{
                (3 * H * D) / SMEM_N_QKV, SMEM_N_QKV / 32, 4, 8, C / SMEM_K, SMEM_K / 64, 4};
        std::vector<int64_t> pmt{0, 4, 1, 5, 3, 2, 6};
        wqkv_sf_fp8 = wqkv_sf_fp8.view(shape).permute(pmt).contiguous();
        at::Tensor wqkv_sf_fp8x4 = wqkv_sf_fp8.view(at::kInt);
        wqkv_sf_fp8x4 = wqkv_sf_fp8x4
                                .view({(3 * H * D) / SMEM_N_QKV, C / SMEM_K,
                                       (SMEM_N_QKV / 32) * (SMEM_K / 64), 32})
                                .contiguous();

        // This is the nvfp4 qkv weights
        wqkv_weights_f16.t = torch::cat({wqkv_fp4, wqkv_sf_fp8x4}, -1);

        //Rotary embedding as a torch tensor
        auto rot_bfrs = self_attn->rotary_emb->named_buffers();
        auto max_T = 16 * (rot_bfrs["sin_freqs"].size(0) / 16);
        sincos_bfr = torch::empty({max_T, D / 2, 2}, f16_opts);
        sincos_bfr.select(2, 0) = rot_bfrs["sin_freqs"].slice(0, 0, max_T).view({max_T, D / 2});
        sincos_bfr.select(2, 1) = rot_bfrs["cos_freqs"].slice(0, 0, max_T).view({max_T, D / 2});
        sincos_bfr =
                sincos_bfr.view({max_T / SMEM_M_QKV, SMEM_M_QKV / 32, 2, 2, 8, D / 32, 2, 4, 2, 2})
                        .permute({0, 1, 5, 2, 6, 4, 7, 8, 3, 9})
                        .contiguous()
                        .reshape({max_T, D});

        // Proj weights
        auto proj_weight_interleaved = self_attn->out_proj->weight.view({C / 64, 4, 8, 2, H * D})
                                               .transpose(1, 2)
                                               .contiguous()
                                               .reshape({C, H * D});
        proj_weight =
                convert_to_nvfp4_tiled(proj_weight_interleaved, {SMEM_N, SMEM_K}, true, i32_opts);

        // Projection Bias weights
        proj_bias = self_attn->out_proj->bias.view({C});

        // Both RMSNorm weights
        t_res_weights = norm1->weight.view({C});
        t_res2_weights = norm2->weight.view({C});

        // SwiGLU
        // Interleave weights for (1) stores and (2) Wx and Wy tiles
        auto fc1_weight_interleaved = ff->fc1->weight.view({E / 64, 4, 8, 2, C})
                                              .transpose(1, 2)
                                              .contiguous()
                                              .reshape({E, C});
        fc1_weight_interleaved = fc1_weight_interleaved.view({2, (E / 2) / 32, 32, C})
                                         .transpose(0, 1)
                                         .contiguous()
                                         .reshape({E, C});

        t_fc1_wts_f16.t =
                convert_to_nvfp4_tiled(fc1_weight_interleaved, {SMEM_N, SMEM_K}, true, i32_opts);

        // FC2
        auto fc2_weights_interleaved = ff->fc2->weight.view({C / 64, 4, 8, 2, E / 2})
                                               .transpose(1, 2)
                                               .contiguous()
                                               .reshape({C, E / 2});
        t_fc2_wts =
                convert_to_nvfp4_tiled(fc2_weights_interleaved, {SMEM_N, SMEM_K}, true, i32_opts);
    }
    utils::ScopedProfileRange layer_spr("TxLayerKoiBlackwell12", 2);

    // Output buffers
    auto qkv = torch::empty({N * T * 3 * H * D}, f8_opts);
    auto t_out_attn_nvfp4 = torch::empty({(N * T) / 256, (H * D) / 128, 8 * 2, 288}, i32_opts);
    auto t_residual_nvfp4 = torch::empty_like(x_tiled_nvfp4);
    auto t_fc1_out_nvfp4 = torch::empty({(N * T) / 256, (E / 2) / 128, 8 * 2, 288}, i32_opts);

    int res = KOI_SUCCESS;
    int calls = 0;
    std::stringstream ss;
    if (res == KOI_SUCCESS && ++calls) {
        // Fused QKV Matmul Plus Rotary embedding
        utils::ScopedProfileRange spr("QKV+ROTE B12", 3);
        res = koi_b12_qkv_rope(stream, x_tiled_nvfp4.data_ptr(), wqkv_weights_f16.t.data_ptr(),
                               sincos_bfr.data_ptr(), qkv.data_ptr(), N * T, C, T);
    }
    if (res == KOI_SUCCESS && ++calls) {
        // Apply window flashattention
        utils::ScopedProfileRange spr("MEA B12", 3);
        res = koi_b12_flashattn(stream, qkv.data_ptr(), t_out_attn_nvfp4.data_ptr(), N, T);
    }
    if (res == KOI_SUCCESS && ++calls) {
        // RMS residual
        utils::ScopedProfileRange spr("Proj + Residual + RMSNorm B12", 3);
        // TODO: Add bias functionality, testing for performance only now...
        res = koi_b12_mm_bias_res_rmsnorm(stream, t_out_attn_nvfp4.data_ptr(),
                                          proj_weight.data_ptr(), t_res_weights.data_ptr(),
                                          proj_bias.data_ptr(),
                                          x_tiled_nvfp4.data_ptr(),     // res
                                          t_residual_nvfp4.data_ptr(),  // output
                                          N * T, C, H * D);
    }
    if (res == KOI_SUCCESS && ++calls) {
        // Matmul + SWIGLU
        utils::ScopedProfileRange spr("FC1+SILU B12", 3);
        res = koi_b12_swiglu(stream, t_residual_nvfp4.data_ptr(), t_fc1_wts_f16.t.data_ptr(),
                             t_fc1_out_nvfp4.data_ptr(), N * T, C, use_uswiglu);
    }
    if (res == KOI_SUCCESS && ++calls) {
        // Koi linear matmul, fc2 weights (K=2048)
        utils::ScopedProfileRange spr("FC2 + Residual + RMSNorm B12", 3);
        res = koi_b12_mm_bias_res_rmsnorm(stream, t_fc1_out_nvfp4.data_ptr(), t_fc2_wts.data_ptr(),
                                          t_res2_weights.data_ptr(), nullptr,
                                          t_residual_nvfp4.data_ptr(),  // res
                                          x_tiled_nvfp4.data_ptr(),     // output
                                          N * T, C, E / 2);
    }
    if (res != KOI_SUCCESS) {
        spdlog::error("Koi B12 tiled path failed {}", calls);
    }
#else
    (void)x_tiled_nvfp4;
    (void)N;
    (void)T;
#endif
}

void TxEncoderImpl::koi_volta_forward(at::Tensor &x_f16) {
    (void)x_f16;
#if DORADO_CUDA_BUILD && !DORADO_ORIN && (CUDA_VERSION / 1000) < 13
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
                            t_fc1_out.data_ptr(), useFloatAccumSwiglu, use_uswiglu, N * T);
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
        DORADO_LOG(error, "Koi Volta tiled path failed {}", calls);
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
#if DORADO_CUDA_BUILD
    // TODO: make sure these are all the requirements
    const auto &layers = params.layer_params;
    const bool is_sup_model =
            (layers.d_model == 512) && (layers.nhead == 8) && (layers.attn_window.first == 127) &&
            (layers.attn_window.second == 128) && (layers.dim_feedforward == 2048);
    const bool is_hyp_model =
            (layers.d_model == 1536) && (layers.nhead == 24) && (layers.attn_window.first == 255) &&
            (layers.attn_window.second == 256) && (layers.dim_feedforward == 6144);
    use_koi_tiled = (is_sup_model || is_hyp_model) && options.device().is_cuda() &&
                    (koi_tc_is_available(KOI_F16) == KOI_SUCCESS) &&
                    utils::get_dev_opt<bool>("use_koi_tiled", true);
    DORADO_LOG(debug, "TxEncoderStack: use_koi_tiled {}.", use_koi_tiled);

#if !DORADO_ORIN
    // Koi kernels support hyp_model, but no testing has been performed, excluding it for now
    use_koi_nvfp4_tiled = is_sup_model && options.device().is_cuda() &&
                          (koi_tc_is_available(KOI_E2M1) == KOI_SUCCESS) &&
                          utils::get_dev_opt<bool>("use_koi_nvfp4_tiled", false);
    spdlog::debug("TxEncoderStack: use_koi_nvfp4_tiled {}.", use_koi_nvfp4_tiled);

#if (CUDA_VERSION / 1000) < 13
    // Custom Volta flag
    use_koi_volta_tiled = is_sup_model && options.device().is_cuda() &&
                          (koi_volta_tc_is_available(KOI_F16) == KOI_SUCCESS);
    DORADO_LOG(debug, "TxEncoderStack: use_koi_volta_tiled {}.", use_koi_volta_tiled);
#endif
#endif
#endif

    const bool use_uswiglu = utils::get_dev_opt("use_uswiglu", false);
    stack = Sequential();
    for (int i = 0; i < params.depth; ++i) {
        TxEncoder encoder(params.layer_params, options, use_uswiglu);
        stack->push_back(register_module("transformer_encoder" + std::to_string(i), encoder));
        layer_vec.push_back(encoder);
    }
    use_i8 = utils::get_dev_opt<bool>("koi_use_i8", true);
};

at::Tensor TxEncoderStackImpl::forward(const at::Tensor &x, [[maybe_unused]] AuxiliaryData *aux) {
#if DORADO_CUDA_BUILD
    if (use_koi_nvfp4_tiled && !DORADO_ORIN) {
#if !DORADO_ORIN
        bool test_nvfp4_tx_perf = utils::get_dev_opt<bool>("koi_test_nvfp4_tx_perf", true);

        if (aux) {
            throw std::runtime_error(
                    "NVFP4 Blackwell inference does not support "
                    "Variable Chunk Sizes yet");
        }

        const int N = static_cast<int>(x.size(0));
        const int T = static_cast<int>(x.size(1));
        const int C = static_cast<int>(x.size(2));

        at::Tensor x_tiled_nvfp4;
        {
            utils::ScopedProfileRange spr("Tile Input to NVFP4", 2);
            if (test_nvfp4_tx_perf) {
                x_tiled_nvfp4 = torch::randn({(N * T) / 256, C / 128, 16, 288},
                                             x.options().dtype(torch::kFloat))
                                        .view(torch::kInt);
            } else {
                x_tiled_nvfp4 = convert_to_nvfp4_tiled(x.reshape({(N * T), C}), {256, 128}, false,
                                                       x.options().dtype(torch::kInt));
            }
        }

        for (auto &layer : layer_vec) {
            layer->koi_nvfp4_forward(x_tiled_nvfp4, N, T);
        }

        at::Tensor untiled_f16;
        {
            utils::ScopedProfileRange spr("Untile NVFP4 Output and Convert to F16", 2);
            if (test_nvfp4_tx_perf) {
                untiled_f16 = x_tiled_nvfp4;
                untiled_f16 = torch::randn({N, T, C}, x.options().dtype(torch::kHalf));
            } else {
                untiled_f16 = untile_nvfp4(x_tiled_nvfp4, {256, 128},
                                           x_tiled_nvfp4.options().dtype(torch::kHalf));
                untiled_f16 = untiled_f16.view({N, T, C});
            }
        }
        return untiled_f16;
#endif  // !DORADO_ORIN
    } else if (use_koi_tiled) {
        int N, T, C;
        if (aux) {
            // N = total amount of chunk_size_granularity within batch
            // T = chunk_size_granularity which is min_chunksize / ConvStack_stride
            N = aux->total_num_granularity();
            T = aux->chunk_size_granularity_tx_enc();
            C = static_cast<int>(x.size(0) *
                                 x.size(2));  // Underlying input layout is (C / 8, M_in, 8)

            // Why do this? Koi's linear.cu MatMulOp implementation requires M to be a multiple of 256
            int p = 256 / T;
            N += ((N % p) == 0) ? 0 : (p - (N % p));
            if (N > aux->max_num_granularity()) {
                throw std::runtime_error(
                        "Tx Encoder total_num_granularity is exceeding "
                        "max_num_granularity\n"
                        "total_num_granularity = " +
                        std::to_string(aux->total_num_granularity()) +
                        ", max_num_granularity = " + std::to_string(aux->max_num_granularity()));
            }
            assert(T == 64);

            auto stream = at::cuda::getCurrentCUDAStream().stream();
            koi_vcs_tx_fill_qkv_rope_lut(stream, aux->total_num_varlen_chunks(), T,
                                         aux->device_chunk_table.data_ptr<int>(),
                                         aux->qkv_rope_lut.data_ptr<int>());
        } else {
            N = static_cast<int>(x.size(0));
            T = static_cast<int>(x.size(1));
            C = static_cast<int>(x.size(2));
        }

        at::Tensor tiled_f16;
        {
            utils::ScopedProfileRange spr("Tile F16", 2);
            if (aux) {
                // Underlying input layout is (C / 8, M_in, 8)
                tiled_f16 = x.narrow(1, 0, N * T)
                                    .view({C / 8, N, T / 16, 16, 8})
                                    .permute({1, 2, 0, 3, 4})
                                    .contiguous();
            } else {
                tiled_f16 = x.view({N, T / 16, 16, C / 8, 8}).transpose(2, 3).contiguous();
            }
        }

        utils::ScaledTensor scaled_tensor;
        if (use_i8) {
            utils::ScopedProfileRange spr("Quantise/tile I8", 2);
            // Quanitze tensor to i8 and take the reciprocal of the scale.
            if (aux) {
                scaled_tensor = utils::quantize_tensor(x.narrow(1, 0, N * T)
                                                               .view({C / 8, N * T, 8})
                                                               .transpose(0, 1)
                                                               .contiguous()
                                                               .view({N * T, C}),
                                                       1);
            } else {
                scaled_tensor = utils::quantize_tensor(x, 2);
            }
            scaled_tensor.scale.reciprocal_();
            // Transform the tensor into the tiled format.
            scaled_tensor.t =
                    scaled_tensor.t.view({N, T / 16, 16, C / 16, 16}).transpose(2, 3).contiguous();
        } else {
            scaled_tensor.t = tiled_f16;
        }

        for (auto &layer : layer_vec) {
            layer->koi_forward(scaled_tensor, tiled_f16, aux);
        }

        at::Tensor untiled_f16;
        {
            utils::ScopedProfileRange spr("Untile F16", 2);
            untiled_f16 = tiled_f16.transpose(2, 3).contiguous().view({N, T, C});
        }
        return untiled_f16;
    }
#if ((CUDA_VERSION / 1000) < 13) && !DORADO_ORIN
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
#endif  // (CUDA_VERSION / 1000) < 13 && !DORADO_ORIN
#endif  // DORADO_CUDA_BUILD
    return stack->forward(x);
}

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

}  // namespace dorado::nn
