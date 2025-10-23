#include "nn/AuxiliaryData.h"

#include "nn/KoiUtils.h"

#include <exception>
#include <numeric>

#if DORADO_CUDA_BUILD
#include <c10/cuda/CUDAStream.h>

extern "C" {
#include "koi.h"
}
#endif

namespace dorado {
namespace nn {

AuxiliaryData::AuxiliaryData(at::Tensor workspace,
                             const std::int32_t batch_size,
                             const std::int32_t chunk_size,
                             const std::int32_t stride,
                             const std::span<const std::int32_t> chunk_sizes)
        : workspace_(std::move(workspace)),
          N_(batch_size),
          T_in_(chunk_size),
          T_out_(chunk_size / stride),
          T_lstm_(1 + T_out_ + 1),
          stride_(stride),
          chunk_sizes_(std::cbegin(chunk_sizes), std::cend(chunk_sizes)) {
    T_lstm_ += T_lstm_ & 1;  // needs to be even for easier LUT creation

    for (std::int32_t & cs : chunk_sizes_) {
        cs /= stride;
    }

    chunk_offsets_ = chunk_sizes_;
    std::exclusive_scan(std::cbegin(chunk_offsets_), std::cend(chunk_offsets_),
                        std::begin(chunk_offsets_), 0);

    const std::int32_t num_chunks = std::ssize(chunk_sizes);
    chunk_intervals_.resize(2 * num_chunks);
    for (std::int32_t i = 0; i < num_chunks; ++i) {
        chunk_intervals_[(2 * i) + 1] = chunk_sizes[i];
    }
    std::partial_sum(std::cbegin(chunk_intervals_), std::cend(chunk_intervals_),
                     std::begin(chunk_intervals_));
}

void AuxiliaryData::create_convolution_auxiliary_data([[maybe_unused]] const torch::Device device) {
#if DORADO_CUDA_BUILD
    if (device_chunk_intervals.defined()) {
        return;
    }

    auto options = at::TensorOptions().dtype(torch::kInt32);

    device_chunk_intervals =
            at::from_blob(std::data(chunk_intervals_),
                          {static_cast<std::int32_t>(std::size(chunk_intervals_))}, options)
                    .to(options.device(device));
#else
    throw std::runtime_error("AuxiliaryData error: unsupported code path!");
#endif
}

void AuxiliaryData::create_lstm_auxiliary_data([[maybe_unused]] const torch::Device device,
                                               [[maybe_unused]] KoiThreads & thread_pool) {
#if DORADO_CUDA_BUILD
    if (device_in_layout.defined()) {
        return;
    }

    auto options = at::TensorOptions().device(device);
    auto stream = c10::cuda::getCurrentCUDAStream(device.index());

    const std::int32_t chunk_sum =
            std::accumulate(std::cbegin(chunk_sizes_), std::cend(chunk_sizes_), 0);

    device_in_layout = at::empty({chunk_sum}, options.dtype(torch::kInt32));
    device_out_layout = at::empty({N_ * (T_lstm_ + 1)}, options.dtype(torch::kInt32));
    device_fwd_encoding = at::empty({N_ * T_lstm_}, options.dtype(torch::kInt32));
    device_bwd_encoding = at::empty({N_ * (T_lstm_ + 1)}, options.dtype(torch::kInt32));

    constexpr std::int32_t SUBBATCH_SIZE{32};

    const int status = host_lstm_preprocess(
            stream.stream(), N_, std::data(chunk_sizes_), std::size(chunk_sizes_), SUBBATCH_SIZE,
            T_lstm_, workspace_.data_ptr<std::int32_t>(), workspace_.size(0), thread_pool.get(),
            nullptr, device_out_layout.data_ptr<std::int32_t>(),
            device_fwd_encoding.data_ptr<std::int32_t>(), device_in_layout.data_ptr<std::int32_t>(),
            nullptr, device_bwd_encoding.data_ptr<std::int32_t>());

    if (status != KOI_SUCCESS) {
        throw std::runtime_error("RNN auxiliary data creation failed.");
    }
#else
    throw std::runtime_error("AuxiliaryData error: unsupported code path!");
#endif
}

void AuxiliaryData::create_decoder_auxiliary_data([[maybe_unused]] const torch::Device device) {
#if DORADO_CUDA_BUILD
    if (device_chunk_sizes.defined()) {
        return;
    }

    auto options = at::TensorOptions().dtype(torch::kInt32);

    device_chunk_sizes =
            at::from_blob(std::data(chunk_sizes_),
                          {static_cast<std::int32_t>(std::size(chunk_sizes_))}, options)
                    .to(options.device(device));

    device_chunk_offsets =
            at::from_blob(std::data(chunk_offsets_),
                          {static_cast<std::int32_t>(std::size(chunk_offsets_))}, options)
                    .to(options.device(device));
#else
    throw std::runtime_error("AuxiliaryData error: unsupported code path!");
#endif
}

}  // namespace nn
}  // namespace dorado
