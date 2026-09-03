#include "flash_attention3_backend.h"

#include <c10/util/Exception.h>

#include <optional>

#if DORADO_HAS_FLASHATTENTION3
#include <ATen/cuda/CUDAContextLight.h>
#include <flash_api.h>
#include <spdlog/spdlog.h>
#endif

namespace dorado::secondary {

namespace {

#if DORADO_HAS_FLASHATTENTION3

bool flash_attention3_device_supported() {
    const cudaDeviceProp* dprops = at::cuda::getCurrentDeviceProperties();
    if (dprops == nullptr) {
        return false;
    }

    const bool supported =
            flash_attention3_compute_capability_supported(dprops->major, dprops->minor);

    if (!supported) {
        static std::once_flag log_once;
        std::call_once(log_once, [major = dprops->major, minor = dprops->minor]() {
            spdlog::info(
                    "FlashAttention3 disabled on CUDA compute capability {}.{}; this build only "
                    "contains {} kernels. Falling back to SDPA.",
                    major, minor, flash_attention3_supported_arch_description());
        });
    }

    return supported;
}
#endif

}  // namespace

bool flash_attention3_available_for(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
#if DORADO_HAS_FLASHATTENTION3
    if (!q.defined() || !k.defined() || !v.defined()) {
        return false;
    }
    if (!q.is_cuda() || !k.is_cuda() || !v.is_cuda()) {
        return false;
    }
    if ((q.scalar_type() != at::ScalarType::Half) &&
        (q.scalar_type() != at::ScalarType::BFloat16)) {
        return false;
    }
    if ((k.scalar_type() != q.scalar_type()) || (v.scalar_type() != q.scalar_type())) {
        return false;
    }
    if ((q.dim() != 4) || (k.dim() != 4) || (v.dim() != 4)) {
        return false;
    }
    if ((q.size(0) != k.size(0)) || (k.size(0) != v.size(0))) {
        return false;
    }
    if (q.size(1) <= 0 || k.size(1) <= 0 || v.size(1) <= 0) {
        return false;
    }
    if ((q.size(2) != k.size(2)) || (k.size(2) != v.size(2))) {
        return false;
    }
    if ((q.size(3) != k.size(3)) || (k.size(3) != v.size(3))) {
        return false;
    }
    if ((q.size(3) <= 0) || (q.size(3) > 256)) {
        return false;
    }
    if ((q.size(3) % 8) != 0) {
        return false;
    }
    if ((q.stride(-1) != 1) || (k.stride(-1) != 1) || (v.stride(-1) != 1)) {
        return false;
    }

    return flash_attention3_device_supported();
#else
    static_cast<void>(q);
    static_cast<void>(k);
    static_cast<void>(v);
    return false;
#endif
}

bool flash_attention3_varlen_available_for(const at::Tensor& q,
                                           const at::Tensor& k,
                                           const at::Tensor& v,
                                           const at::Tensor& cumsum_seqlens_q,
                                           const at::Tensor& cumsum_seqlens_k) {
#if DORADO_HAS_FLASHATTENTION3
    if (!q.defined() || !k.defined() || !v.defined() || !cumsum_seqlens_q.defined() ||
        !cumsum_seqlens_k.defined()) {
        return false;
    }
    if (!q.is_cuda() || !k.is_cuda() || !v.is_cuda() || !cumsum_seqlens_q.is_cuda() ||
        !cumsum_seqlens_k.is_cuda()) {
        return false;
    }
    if ((q.scalar_type() != at::ScalarType::Half) &&
        (q.scalar_type() != at::ScalarType::BFloat16)) {
        return false;
    }
    if ((k.scalar_type() != q.scalar_type()) || (v.scalar_type() != q.scalar_type())) {
        return false;
    }
    if ((q.dim() != 3) || (k.dim() != 3) || (v.dim() != 3)) {
        return false;
    }
    if (k.size(0) != v.size(0)) {
        return false;
    }
    if (q.size(0) <= 0 || k.size(0) <= 0 || v.size(0) <= 0) {
        return false;
    }
    if ((q.size(1) != k.size(1)) || (k.size(1) != v.size(1))) {
        return false;
    }
    if ((q.size(2) != k.size(2)) || (k.size(2) != v.size(2))) {
        return false;
    }
    if ((q.size(2) <= 0) || (q.size(2) > 256)) {
        return false;
    }
    if ((q.size(2) % 8) != 0) {
        return false;
    }
    if ((q.stride(-1) != 1) || (k.stride(-1) != 1) || (v.stride(-1) != 1)) {
        return false;
    }

    if ((cumsum_seqlens_q.dim() != 1) || (cumsum_seqlens_k.dim() != 1)) {
        return false;
    }
    if (cumsum_seqlens_q.size(0) != cumsum_seqlens_k.size(0)) {
        return false;
    }
    if ((cumsum_seqlens_q.scalar_type() != at::ScalarType::Int)) {
        return false;
    }
    if ((cumsum_seqlens_k.scalar_type() != at::ScalarType::Int)) {
        return false;
    }

    return flash_attention3_device_supported();
#else
    static_cast<void>(q);
    static_cast<void>(k);
    static_cast<void>(v);
    static_cast<void>(cumsum_seqlens_q);
    static_cast<void>(cumsum_seqlens_k);
    return false;
#endif
}

at::Tensor flash_attention3_forward(const at::Tensor& q,
                                    const at::Tensor& k,
                                    const at::Tensor& v,
                                    const std::optional<at::Tensor>& cumsum_seqlens_q,
                                    const std::optional<at::Tensor>& cumsum_seqlens_k) {
#if DORADO_HAS_FLASHATTENTION3
    std::optional<int64_t> max_seqlen_q{std::nullopt};
    if (cumsum_seqlens_q) {
        // The .diff(0) computes the discrete difference along dimension 0.
        max_seqlen_q = {(*cumsum_seqlens_q).diff(0).max().item<int64_t>()};
    }
    std::optional<int64_t> max_seqlen_k{std::nullopt};
    if (cumsum_seqlens_k) {
        // The .diff(0) computes the discrete difference along dimension 0.
        max_seqlen_k = {(*cumsum_seqlens_k).diff(0).max().item<int64_t>()};
    }

    auto [out, softmax_lse, out_accum, softmax_lse_accum] = ::mha_fwd(
            q, k, v, std::nullopt, std::nullopt, std::nullopt, std::nullopt, cumsum_seqlens_q,
            cumsum_seqlens_k, std::nullopt, std::nullopt, std::nullopt, max_seqlen_q, max_seqlen_k,
            std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
            std::nullopt, std::nullopt, std::nullopt, std::nullopt,
            /*is_causal=*/false,
            /*window_size_left=*/-1,
            /*window_size_right=*/-1,
            /*attention_chunk=*/0,
            /*softcap=*/0.0,
            /*is_rotary_interleaved=*/false, std::nullopt,
            /*num_splits=*/0, std::nullopt,
            /*sm_margin=*/0);
    static_cast<void>(softmax_lse);
    static_cast<void>(out_accum);
    static_cast<void>(softmax_lse_accum);
    return out;
#else
    static_cast<void>(q);
    static_cast<void>(k);
    static_cast<void>(v);
    static_cast<void>(cumsum_seqlens_q);
    static_cast<void>(cumsum_seqlens_k);
    TORCH_CHECK(false, "FlashAttention3 is not available in this build.");
#endif
}

}  // namespace dorado::secondary
