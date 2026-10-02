#include "flash_attention_module.h"

#include "flash_attention3_backend.h"
#include "torch_utils/gpu_profiling.h"
#include "torch_utils/tensor_utils.h"

#include <ATen/ops/scaled_dot_product_attention.h>
#include <c10/util/Exception.h>
#include <spdlog/spdlog.h>
#include <torch/headeronly/core/ScalarType.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <vector>

#if DORADO_CUDA_BUILD
#include <ATen/cuda/CUDAContextLight.h>
#endif

namespace dorado::secondary {

namespace {

/**
 * \brief Check whether the current CUDA device supports BF16 attention execution.
 * \returns True when running on a CUDA build with a compute capability 8.x or newer device.
 */
bool cuda_bf16_attention_supported() {
#if DORADO_CUDA_BUILD
    const auto* dprops = at::cuda::getCurrentDeviceProperties();
    return (dprops != nullptr) && (dprops->major >= 8);
#else
    return false;
#endif
}

/**
 * \brief Decide whether the attention path should cast FP32 CUDA inputs to BF16.
 * \param q Query tensor.
 * \param k Key tensor.
 * \param v Value tensor.
 * \returns True when all inputs are CUDA FP32 tensors and the current device supports BF16
 *          attention execution.
 */
bool use_attention_bf16_by_default(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
    if (!q.is_cuda() || !k.is_cuda() || !v.is_cuda()) {
        return false;
    }
    if ((q.scalar_type() != at::ScalarType::Float) || (k.scalar_type() != at::ScalarType::Float) ||
        (v.scalar_type() != at::ScalarType::Float)) {
        return false;
    }
    if (!cuda_bf16_attention_supported()) {
        return false;
    }
    return true;
}

/**
 * \brief Run the SDPA fallback for batched attention inputs.
 * \param q Query tensor of shape (batch_size, query_tokens, num_heads, head_dim).
 * \param k Key tensor of shape (batch_size, key_tokens, num_heads, head_dim).
 * \param v Value tensor of shape (batch_size, key_tokens, num_heads, head_dim).
 * \param attn_mask Optional attention mask broadcastable to
 *        (batch_size, num_heads, query_tokens, key_tokens).
 * \returns Attention output tensor with the same shape as \p q.
 */
at::Tensor scaled_dot_product_attention_batched_fallback(
        const at::Tensor& q,
        const at::Tensor& k,
        const at::Tensor& v,
        const c10::optional<at::Tensor>& attn_mask) {
    const at::Tensor q2 = q.permute({0, 2, 1, 3});
    const at::Tensor k2 = k.permute({0, 2, 1, 3});
    const at::Tensor v2 = v.permute({0, 2, 1, 3});
    const at::Tensor attn = at::scaled_dot_product_attention(q2, k2, v2, attn_mask);
    return attn.permute({0, 2, 1, 3}).contiguous();
}

/**
 * \brief Run the SDPA fallback for packed variable-length attention inputs.
 * \param q Packed query tensor of shape (total_query_tokens, num_heads, head_dim).
 * \param k Packed key tensor of shape (total_key_tokens, num_heads, head_dim).
 * \param v Packed value tensor of shape (total_key_tokens, num_heads, head_dim).
 * \param cumsum_seqlens_q Prefix-sum sequence boundaries for \p q with shape (batch_size + 1).
 * \param cumsum_seqlens_k Prefix-sum sequence boundaries for \p k and \p v with shape
 *        (batch_size + 1).
 * \returns Attention output with the same packed layout and shape as \p q.
 */
at::Tensor scaled_dot_product_attention_varlen_fallback(const at::Tensor& q,
                                                        const at::Tensor& k,
                                                        const at::Tensor& v,
                                                        const at::Tensor& cumsum_seqlens_q,
                                                        const at::Tensor& cumsum_seqlens_k) {
    TORCH_CHECK(cumsum_seqlens_q.dim() == 1, "cumsum_seqlens_q must be 1D");
    TORCH_CHECK(cumsum_seqlens_k.dim() == 1, "cumsum_seqlens_k must be 1D");
    TORCH_CHECK(cumsum_seqlens_q.size(0) == cumsum_seqlens_k.size(0),
                "cumsum_seqlens_q and cumsum_seqlens_k must have the same length");

    const at::Tensor cumsum_seqlens_q_cpu = cumsum_seqlens_q.to(torch::kCPU).contiguous();
    const at::Tensor cumsum_seqlens_k_cpu = cumsum_seqlens_k.to(torch::kCPU).contiguous();
    const auto* q_offsets = cumsum_seqlens_q_cpu.data_ptr<int32_t>();
    const auto* k_offsets = cumsum_seqlens_k_cpu.data_ptr<int32_t>();
    const int64_t batch_size = cumsum_seqlens_q_cpu.size(0) - 1;

    at::Tensor out = at::zeros_like(q);
    std::vector<int64_t> active_q_starts;
    std::vector<int64_t> active_q_lens;
    std::vector<int64_t> active_k_starts;
    std::vector<int64_t> active_k_lens;
    active_q_starts.reserve(batch_size);
    active_q_lens.reserve(batch_size);
    active_k_starts.reserve(batch_size);
    active_k_lens.reserve(batch_size);

    int64_t max_q_len = 0;
    int64_t max_k_len = 0;
    for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const int64_t q_begin = q_offsets[batch_idx];
        const int64_t q_end = q_offsets[batch_idx + 1];
        const int64_t k_begin = k_offsets[batch_idx];
        const int64_t k_end = k_offsets[batch_idx + 1];
        const int64_t q_len = q_end - q_begin;
        const int64_t k_len = k_end - k_begin;

        if ((q_len <= 0) || (k_len <= 0)) {
            continue;
        }

        active_q_starts.push_back(q_begin);
        active_q_lens.push_back(q_len);
        active_k_starts.push_back(k_begin);
        active_k_lens.push_back(k_len);
        max_q_len = std::max(max_q_len, q_len);
        max_k_len = std::max(max_k_len, k_len);
    }

    if (active_q_lens.empty()) {
        return out;
    }

    const int64_t active_batch_size = static_cast<int64_t>(active_q_lens.size());
    const int64_t num_heads = q.size(1);
    const int64_t head_dim = q.size(2);
    at::Tensor q_padded =
            at::zeros({active_batch_size, max_q_len, num_heads, head_dim}, q.options());
    at::Tensor k_padded =
            at::zeros({active_batch_size, max_k_len, num_heads, head_dim}, k.options());
    at::Tensor v_padded =
            at::zeros({active_batch_size, max_k_len, num_heads, head_dim}, v.options());

    for (int64_t packed_idx = 0; packed_idx < active_batch_size; ++packed_idx) {
        const int64_t q_begin = active_q_starts[packed_idx];
        const int64_t q_len = active_q_lens[packed_idx];
        const int64_t k_begin = active_k_starts[packed_idx];
        const int64_t k_len = active_k_lens[packed_idx];

        q_padded.select(/*dim=*/0, /*index=*/packed_idx)
                .narrow(/*dim=*/0, /*start=*/0, /*length=*/q_len)
                .copy_(q.narrow(/*dim=*/0, /*start=*/q_begin, /*length=*/q_len));
        k_padded.select(/*dim=*/0, /*index=*/packed_idx)
                .narrow(/*dim=*/0, /*start=*/0, /*length=*/k_len)
                .copy_(k.narrow(/*dim=*/0, /*start=*/k_begin, /*length=*/k_len));
        v_padded.select(/*dim=*/0, /*index=*/packed_idx)
                .narrow(/*dim=*/0, /*start=*/0, /*length=*/k_len)
                .copy_(v.narrow(/*dim=*/0, /*start=*/k_begin, /*length=*/k_len));
    }

    const auto device = q.device();
    const auto lens_opts_cpu = at::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
    const auto mask_opts = at::TensorOptions().dtype(torch::kLong).device(device);
    const at::Tensor q_lens = at::tensor(active_q_lens, lens_opts_cpu).to(device);
    const at::Tensor k_lens = at::tensor(active_k_lens, lens_opts_cpu).to(device);
    const at::Tensor q_positions = at::arange(max_q_len, mask_opts).unsqueeze(0);
    const at::Tensor k_positions = at::arange(max_k_len, mask_opts).unsqueeze(0);
    const at::Tensor q_valid = q_positions.lt(q_lens.unsqueeze(1));
    const at::Tensor k_valid = k_positions.lt(k_lens.unsqueeze(1));
    const at::Tensor attn_mask =
            q_valid.unsqueeze(-1).logical_and(k_valid.unsqueeze(-2)).unsqueeze(1);

    const at::Tensor out_padded =
            scaled_dot_product_attention_batched_fallback(q_padded, k_padded, v_padded, attn_mask);

    for (int64_t packed_idx = 0; packed_idx < active_batch_size; ++packed_idx) {
        const int64_t q_begin = active_q_starts[packed_idx];
        const int64_t q_len = active_q_lens[packed_idx];
        out.narrow(/*dim=*/0, /*start=*/q_begin, /*length=*/q_len)
                .copy_(out_padded.select(/*dim=*/0, /*index=*/packed_idx)
                               .narrow(/*dim=*/0, /*start=*/0, /*length=*/q_len));
    }

    return out;
}

}  // namespace

at::Tensor FlashAttentionModuleImpl::forward_batched(const at::Tensor& q,
                                                     const at::Tensor& k,
                                                     const at::Tensor& v) const {
    return this->forward(q, k, v, std::nullopt, std::nullopt);
}

at::Tensor FlashAttentionModuleImpl::forward_varlen(const at::Tensor& q,
                                                    const at::Tensor& k,
                                                    const at::Tensor& v,
                                                    const at::Tensor& cumsum_seqlens_q,
                                                    const at::Tensor& cumsum_seqlens_k) const {
    return this->forward(q, k, v, cumsum_seqlens_q, cumsum_seqlens_k);
}

at::Tensor FlashAttentionModuleImpl::forward(
        const at::Tensor& q,
        const at::Tensor& k,
        const at::Tensor& v,
        const std::optional<at::Tensor>& cumsum_seqlens_q,
        const std::optional<at::Tensor>& cumsum_seqlens_k) const {
    utils::ScopedProfileRange spr("FlashAttentionModuleImpl::forward", 4);

    if (!q.defined() || !k.defined() || !v.defined()) {
        throw std::runtime_error{"FlashAttentionModuleImpl: q/k/v must be defined tensors."};
    }
    if ((cumsum_seqlens_q && !(cumsum_seqlens_k)) || (!(cumsum_seqlens_q) && cumsum_seqlens_k)) {
        throw std::runtime_error{
                "FlashAttentionModuleImpl: inconsistent specification of cumsum_seqlens_q/k."};
    }
    if (k.sizes() != v.sizes()) {
        throw std::runtime_error{"FlashAttentionModuleImpl: k and v shapes must match. k.shape = " +
                                 utils::tensor_shape_as_string(k) +
                                 ", v.shape = " + utils::tensor_shape_as_string(v)};
    }

    const bool use_varlen = (cumsum_seqlens_q.has_value());

    if (use_varlen) {
        if ((q.dim() != 3) || (k.dim() != 3) || (v.dim() != 3)) {
            throw std::runtime_error{
                    "FlashAttentionModuleImpl (varlen): q/k/v tensors must be 3D. q.shape = " +
                    utils::tensor_shape_as_string(q)};
        }
        if ((q.size(1) != k.size(1)) || (q.size(2) != k.size(2))) {
            throw std::runtime_error{
                    "FlashAttentionModuleImpl (varlen): q/k tensors must match in H,D dimensions. "
                    "q.shape "
                    "= " +
                    utils::tensor_shape_as_string(q) +
                    ", k.shape = " + utils::tensor_shape_as_string(k)};
        }
    } else {
        if ((q.dim() != 4) || (k.dim() != 4) || (v.dim() != 4)) {
            throw std::runtime_error{
                    "FlashAttentionModuleImpl: q/k/v tensors must be 4D. q.shape = " +
                    utils::tensor_shape_as_string(q)};
        }
        if ((q.size(0) != k.size(0)) || (k.size(0) != v.size(0)) || (q.size(2) != k.size(2)) ||
            (k.size(2) != v.size(2)) || (q.size(3) != k.size(3)) || (k.size(3) != v.size(3))) {
            throw std::runtime_error{
                    "FlashAttentionModuleImpl: q/k/v tensors must match in batch, heads and head "
                    "dimension. q.shape = " +
                    utils::tensor_shape_as_string(q) +
                    ", k.shape = " + utils::tensor_shape_as_string(k) +
                    ", v.shape = " + utils::tensor_shape_as_string(v)};
        }
    }

    at::Tensor q_work = q;
    at::Tensor k_work = k;
    at::Tensor v_work = v;

    // BF16 will only be used if the input is FP32 (not FP16).
    const bool use_attention_bf16 = use_attention_bf16_by_default(q, k, v);

    if (use_attention_bf16) {
        q_work = q.to(at::ScalarType::BFloat16);
        k_work = k.to(at::ScalarType::BFloat16);
        v_work = v.to(at::ScalarType::BFloat16);
        static std::once_flag logged_bf16_cast;
        std::call_once(logged_bf16_cast, []() {
            DORADO_LOG(trace, "FlashAttentionModuleImpl: forward using BF16 on CUDA.");
        });
    }

    bool use_flashattention = false;

    if (use_varlen) {
        use_flashattention = flash_attention3_varlen_available_for(
                q_work, k_work, v_work, (*cumsum_seqlens_q), (*cumsum_seqlens_k));
    } else {
        use_flashattention = flash_attention3_available_for(q_work, k_work, v_work);
    }

    at::Tensor out;

    if (use_flashattention) {
        static std::once_flag logged_flashattention_path;
        std::call_once(logged_flashattention_path, [use_varlen]() {
            DORADO_LOG(trace, "Using FlashAttention ({}) attention).",
                       (use_varlen ? "varlen" : "batched"));
        });

        // Note: Intentionally not catching exceptions in case FlashAttention is impure and leaves
        // some global state touched. In this case, propagate the exception upstream to a decision point.
        out = flash_attention3_forward(q_work, k_work, v_work, cumsum_seqlens_q, cumsum_seqlens_k);
    } else {
        static std::once_flag logged_flashattention_unavailable;
        std::call_once(logged_flashattention_unavailable, [use_varlen]() {
            DORADO_LOG(trace,
                       "FlashAttention not available for {} attention. Falling back to "
                       "SDPA.",
                       (use_varlen ? "varlen" : "batched"));
        });

        out = use_varlen ? scaled_dot_product_attention_varlen_fallback(
                                   q_work, k_work, v_work, *cumsum_seqlens_q, *cumsum_seqlens_k)
                         : scaled_dot_product_attention_batched_fallback(q_work, k_work, v_work,
                                                                         std::nullopt);
    }

    if (out.scalar_type() != q.scalar_type()) {
        out = out.to(q.scalar_type());
    }

    return out;
}

}  // namespace dorado::secondary
