#include "ModBaseLatentSpaceV1SequenceModel.h"

#include <ATen/ops/arange.h>

#include <stdexcept>

namespace dorado::modbase::model {

ModBaseLatentSpaceV1SequenceModelImpl::ModBaseLatentSpaceV1SequenceModelImpl(
        const config::modbase::LatentSpaceV1ModelConfig& config)
        : model_config(config) {}

at::Tensor ModBaseLatentSpaceV1SequenceModelImpl::forward(const at::Tensor& decoded_data) {
    if (decoded_data.dim() != 3 || decoded_data.size(0) != 3) {
        throw std::runtime_error(
                "ModBaseLatentSpaceV1SequenceModel input must have shape [3, N, T]");
    }
    if (decoded_data.scalar_type() != torch::kInt8) {
        throw std::runtime_error("ModBaseLatentSpaceV1SequenceModel input must have int8 dtype");
    }

    const auto batch_size = decoded_data.size(1);
    const auto time_steps = decoded_data.size(2);
    if (time_steps == 0) {
        throw std::runtime_error(
                "ModBaseLatentSpaceV1SequenceModel input must have a non-empty time dimension");
    }

    // Decoder::prepare_decode stores the compact sequence at the beginning of the sequence row. Use
    // the cumulative move count to recover the sequence index active at each decoder timestep.
    const auto moves = decoded_data.select(0, 0).to(torch::kLong);
    const auto sequence = decoded_data.select(0, 1);
    const auto cumulative_moves = moves.cumsum(-1);
    const auto current_base = cumulative_moves - 1;
    const auto num_bases = cumulative_moves.select(1, time_steps - 1).reshape({batch_size, 1, 1});

    const auto half_kmer = model_config.general.kmer_len / 2;
    const auto index_options = decoded_data.options().dtype(torch::kLong);
    const auto kmer_offsets =
            at::arange(-half_kmer, half_kmer + 1, 1, index_options).reshape({1, 1, -1});
    const auto kmer_indices = current_base.unsqueeze(-1) + kmer_offsets;

    // Timesteps preceding the first move are padding rather than partial future k-mers. At the
    // sequence ends, out-of-range k-mer positions encode N as four zeroes.
    const auto valid_indices =
            current_base.unsqueeze(-1).ge(0) & kmer_indices.ge(0) & kmer_indices.lt(num_bases);
    const auto safe_indices = kmer_indices.clamp(0, time_steps - 1);

    // Gather compact sequence elements without materialising an N x T x T expanded sequence.
    const auto batch_offsets =
            at::arange(batch_size, index_options).reshape({batch_size, 1, 1}) * time_steps;
    const auto flat_indices = (safe_indices + batch_offsets).reshape({-1});
    const auto kmer_bases =
            sequence.reshape({-1})
                    .index_select(0, flat_indices)
                    .reshape({batch_size, time_steps, model_config.general.kmer_len});

    // Base order matches the existing modbase encoder: A, C, G, T. Unknown characters and N are
    // represented by four zeroes.
    auto encoded = torch::stack(
            {kmer_bases.eq('A'), kmer_bases.eq('C'), kmer_bases.eq('G'), kmer_bases.eq('T')}, -1);
    encoded = (encoded & valid_indices.unsqueeze(-1))
                      .to(torch::kInt8)
                      .reshape({batch_size, time_steps, model_config.general.kmer_len * 4});

    // Decoder output is at the signal-branch stride. Repeating each encoding restores the input
    // resolution expected by a sequence convolution stack whose total stride is sequence_stride.
    if (model_config.sequence_stride > 1) {
        encoded = encoded.repeat_interleave(model_config.sequence_stride, 1);
    }
    return encoded;
}

}  // namespace dorado::modbase::model
