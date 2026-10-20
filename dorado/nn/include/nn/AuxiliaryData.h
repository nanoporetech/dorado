#pragma once

#include <ATen/core/TensorBody.h>
#include <c10/core/Device.h>

#include <cstdint>
#include <span>
#include <vector>

namespace dorado::nn {

class KoiThreads;

class AuxiliaryData {
public:
    AuxiliaryData(at::Tensor workspace,
                  std::int32_t batch_size,
                  std::int32_t chunk_size,
                  std::int32_t stride_out,
                  std::int32_t stride_in,
                  std::int32_t chunk_size_granularity_,
                  std::span<const std::int32_t> chunk_sizes,
                  const std::int32_t max_chunk_size,
                  bool is_tx_model);

    std::int32_t N() const { return N_; }
    std::int32_t T_in() const { return T_in_; }
    std::int32_t T_out() const { return T_out_; }
    std::int32_t T_lstm() const { return T_lstm_; }
    std::int32_t NT_in() const { return chunk_intervals_.back(); }
    std::int32_t NT_out() const { return NT_in() / stride_out_; }
    std::int32_t NT_in_max() const { return N_ * T_in_; }
    std::int32_t NT_out_max() const { return NT_in_max() / stride_out_; }

    void restore_convolution_auxiliary_data();

    bool is_tx_model() const { return is_tx_model_; }
    bool is_lstm_or_flstm_model() const { return !is_tx_model_; }
    std::int32_t chunk_size_granularity() const { return chunk_size_granularity_; }
    std::int32_t total_num_granularity() const { return total_num_granularity_; }
    std::int32_t total_num_varlen_chunks() const { return total_num_varlen_chunks_; }
    std::int32_t max_num_granularity() const { return max_num_granularity_; }
    std::int32_t max_chunk_size_tx_enc() const { return max_chunk_size_ / stride_in_; }
    std::int32_t chunk_size_granularity_tx_enc() const { return chunk_size_granularity_tx_enc_; }
    std::int32_t stride_in() const { return stride_in_; }
    std::int32_t stride_out() const { return stride_out_; }
    void apply_stride_to_chunk_size_granularity(std::int32_t stride) {
        chunk_size_granularity_ /= stride;
    }
    void set_chunk_size_granularity(std::int32_t csg) { chunk_size_granularity_ = csg; }

    void create_auxiliary_data(const c10::Device& device, KoiThreads& thread_pool);

    at::Tensor device_in_layout;
    at::Tensor device_out_layout;
    at::Tensor device_fwd_encoding;
    at::Tensor device_bwd_encoding;

    std::span<const std::int32_t> chunk_sizes() const { return chunk_sizes_; }

    at::Tensor device_chunk_intervals;
    at::Tensor
            device_chunk_table;  // shape = (total_num_varlen_chunks, 2). Column 0 is chunk_start. Column 1 is chunk_length
    at::Tensor conv_load_lut;   // shape = (total_num_granularity_,)
    at::Tensor conv_store_lut;  // shape = (total_num_granularity_,)
    at::Tensor
            qkv_rope_lut;  // shape = (total_num_granularity_,) padded to be multiple of 4, explained in .cpp

private:
    at::Tensor workspace_;
    std::int32_t N_{0};
    std::int32_t T_in_{0};
    std::int32_t T_out_{0};
    std::int32_t T_lstm_{0};
    std::int32_t stride_out_{0};
    std::int32_t stride_in_{0};
    std::vector<std::int32_t> chunk_sizes_;
    std::vector<std::int32_t> chunk_table_;
    std::vector<std::int32_t> chunk_intervals_;

    std::int32_t
            chunk_size_granularity_;  // chunk_size_granularity() from BasecallModelConfig.h, gets modified in ConvStack.cpp
    std::int32_t chunk_size_granularity_tx_enc_;  // Useful to have
    std::int32_t total_num_granularity_;          // Input length divided by chunk_size_granularity
    std::int32_t total_num_varlen_chunks_;        // Amount of varlen chunks in batch
    std::int32_t
            max_num_granularity_;  // Given CudaModel's batch_size and chunk_size, max amount of granularity chunks
    std::int32_t max_chunk_size_;  // Acquired from model's config "[basecaller] chunksize"

    bool is_tx_model_;
};

}  // namespace dorado::nn
