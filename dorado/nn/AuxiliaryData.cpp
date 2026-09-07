#include "nn/AuxiliaryData.h"

#include "nn/KoiThreads.h"

#include <stdexcept>

#if DORADO_CUDA_BUILD
#include "utils/math_utils.h"

#include <ATen/ops/empty.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/zeros.h>
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
                             const std::int32_t stride_out,
                             const std::int32_t stride_in,
                             const std::int32_t chunk_size_granularity,
                             const std::span<const std::int32_t> chunk_sizes,
                             const std::int32_t max_chunk_size,
                             const bool is_tx_model)
        : workspace_(std::move(workspace)),
          N_(batch_size),
          T_in_(chunk_size),
          T_out_(chunk_size / stride_out),
          T_lstm_(1 + T_out_ + 1),
          stride_out_(stride_out),
          stride_in_(stride_in),
          chunk_sizes_(std::cbegin(chunk_sizes), std::cend(chunk_sizes)),
          chunk_size_granularity_(chunk_size_granularity),
          max_chunk_size_(max_chunk_size),
          is_tx_model_(is_tx_model) {
    T_lstm_ += T_lstm_ & 1;  // needs to be even for easier LUT creation

    total_num_varlen_chunks_ = std::ssize(chunk_sizes_);
    chunk_table_.resize(2 * total_num_varlen_chunks_);
    chunk_intervals_.resize(2 * total_num_varlen_chunks_);
    int i = 0;
    total_num_granularity_ = 0;
    for (std::int32_t& cs : chunk_sizes_) {
        int cs_blocks = cs / chunk_size_granularity_;
        chunk_table_[(2 * i) + 0] = total_num_granularity_;
        chunk_table_[(2 * i) + 1] = cs_blocks;
        chunk_intervals_[(2 * i) + 0] = total_num_granularity_ * chunk_size_granularity_;
        chunk_intervals_[(2 * i) + 1] =
                (total_num_granularity_ + cs_blocks) * chunk_size_granularity_;
        total_num_granularity_ += cs_blocks;
        ++i;
        cs /= stride_out_;
    }
}

void AuxiliaryData::restore_convolution_auxiliary_data() {
#if DORADO_CUDA_BUILD
    if (!device_chunk_intervals.defined()) {
        throw std::runtime_error("AuxiliaryData error: undefined chunk intervals!");
    }
    device_chunk_intervals.mul_(stride_in_);
#else
    throw std::runtime_error("AuxiliaryData error: unsupported code path!");
#endif
}

void AuxiliaryData::create_auxiliary_data([[maybe_unused]] const c10::Device& device,
                                          [[maybe_unused]] KoiThreads& thread_pool) {
#if DORADO_CUDA_BUILD

    auto cpu_options = at::TensorOptions().dtype(at::kInt);
    auto gpu_options = cpu_options.device(device);

    if (is_lstm_or_flstm_model()) {
        if (device_in_layout.defined()) {
            return;
        }

        auto stream = c10::cuda::getCurrentCUDAStream(device.index());

        const std::int32_t chunk_sum =
                std::accumulate(std::cbegin(chunk_sizes_), std::cend(chunk_sizes_), 0);

        device_in_layout = at::empty({chunk_sum}, gpu_options);
        device_out_layout = at::empty({N_ * (T_lstm_ + 1)}, gpu_options);
        device_fwd_encoding = at::empty({N_ * T_lstm_}, gpu_options);
        device_bwd_encoding = at::empty({N_ * (T_lstm_ + 1)}, gpu_options);

        constexpr std::int32_t SUBBATCH_SIZE{32};

        const int status = host_lstm_preprocess(
                stream.stream(), N_, std::data(chunk_sizes_), std::size(chunk_sizes_),
                SUBBATCH_SIZE, T_lstm_, workspace_.data_ptr<std::int32_t>(), workspace_.size(0),
                thread_pool.get(), nullptr, device_out_layout.data_ptr<std::int32_t>(),
                device_fwd_encoding.data_ptr<std::int32_t>(),
                device_in_layout.data_ptr<std::int32_t>(), nullptr,
                device_bwd_encoding.data_ptr<std::int32_t>());

        if (status != KOI_SUCCESS) {
            throw std::runtime_error("RNN auxiliary data creation failed.");
        }

        device_chunk_intervals =
                at::from_blob(std::data(chunk_intervals_),
                              {static_cast<std::int32_t>(std::size(chunk_intervals_))}, cpu_options)
                        .to(gpu_options);

        device_chunk_table =
                at::from_blob(std::data(chunk_table_), {total_num_varlen_chunks_, 2}, cpu_options)
                        .to(gpu_options);
    } else {
        if (conv_load_lut.defined() || conv_store_lut.defined() || qkv_rope_lut.defined()) {
            throw std::runtime_error(
                    "We are trying to re-instantiate Tx VCS LUTs, this shouldn't happen! "
                    "total_num_granularity depends on current_batch from BasecallerNode.cpp, and "
                    "we should instantiate new AuxiliaryData with every basecall_current_batch, "
                    "and subsequent call_chunks.");
            return;
        }

        max_num_granularity_ = NT_in_max() / chunk_size_granularity_;
        chunk_size_granularity_tx_enc_ = chunk_size_granularity_ / stride_in_;

        conv_load_lut = at::empty({total_num_granularity_}, gpu_options);
        conv_store_lut = at::empty({total_num_granularity_}, gpu_options);

        // Koi's MatMulOp requires inputs to be a multiple of 256
        int p = 256 / chunk_size_granularity_tx_enc_;
        int qkv_rope_lut_size = utils::pad_to(total_num_granularity_, p);
        qkv_rope_lut = at::zeros({qkv_rope_lut_size}, gpu_options);

        // Why is qkv_rope_lut intialised with zeros?
        // qkv_rope gets sincos values according to T value within lut
        // Input to tx encoder is "padded" to be multiple of 256 to accommodate Koi's MatMulOp
        // So blocks that are outside of total_num_granularity will access lut
        // Those blocks are irrelevant for basecalling, just don't make basecalling crash
        // So with lut being zeros, these padded blocks will calculate sincos as if they were T = 0

        device_chunk_table =
                at::from_blob(std::data(chunk_table_), {total_num_varlen_chunks_, 2}, cpu_options)
                        .mul_(chunk_size_granularity_)
                        .to(gpu_options);
    }
#else
    throw std::runtime_error("AuxiliaryData error: unsupported code path!");
#endif
}

}  // namespace nn
}  // namespace dorado
