#include "CUDADecoder.h"

#include "torch_utils/cuda_utils.h"
#include "torch_utils/gpu_profiling.h"
#include "utils/sequence_utils.h"

#include <ATen/TensorIndexing.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDAGuard.h>
#include <nvtx3/nvtx3.hpp>

#include <numeric>

extern "C" {
#include "koi.h"
}

namespace dorado::basecall::decode {

CUDADecoder::CUDADecoder(const float score_range,
                         const std::span<const config::ModBaseModelConfig> modbase_configs)
        : m_modbase_base_offsets{config::get_modbase_info(modbase_configs).base_probs_offsets()},
          m_score_range{score_range} {
    for (const config::ModBaseModelConfig &config : modbase_configs) {
        const config::modbase::ModBaseModelConfigData &modbase_data{
                config::get_model_config_data(config)};
        const config::modbase::CommonModelParams &common_params{
                config::get_common_model_params(config)};
        m_modbase_decoders.emplace_back(modbase_data.mods.base_id, modbase_data.mods.count + 1,
                                        modbase_data.context.samples_before / common_params.stride,
                                        modbase_data.context.samples_after / common_params.stride,
                                        modbase::MotifMatcher{config});
        m_modbase_num_states += modbase_data.mods.count;
    }
}

DecoderResult CUDADecoder::prepare_decode(DecoderInput input) const {
    auto scores = input.scores;
    if (scores.dtype() == at::kChar && utils::get_dev_opt("convert_scores_f16", false)) {
        utils::ScopedProfileRange spr{"convert_scores", 2};
        scores = scores.to(at::kHalf).mul_(m_score_range / 127.f);
    }

    const auto &options = input.options;

    c10::cuda::CUDAGuard device_guard(scores.device());
    utils::ScopedProfileRange loop{"gpu_decode", 1};
    long int N = input.aux ? std::ssize(input.aux->chunk_sizes()) : (long int)(scores.sizes()[0]);
    long int T = (long int)(scores.sizes()[1]);
    long int C = (long int)(scores.sizes()[2]);

    auto tensor_options_int32 =
            at::TensorOptions().dtype(at::kInt).device(scores.device()).requires_grad(false);
    auto tensor_options_int8 = tensor_options_int32.dtype(at::kChar);

    at::Tensor chunks;
    at::Tensor chunk_results;
    if (input.aux) {
        std::int32_t N_;
        if (input.aux->is_lstm_or_flstm_model()) {
            N_ = std::max<std::int32_t>(N, input.aux->N() * 4);
        } else {
            // N IS TOTAL_NUM_VARLEN_CHUNKS
            N_ = N;
            input.aux->device_chunk_table.floor_divide_(input.aux->stride_out());
        }
        chunks = at::empty({N_, 4}, tensor_options_int32);
        auto chunks_slice = chunks.slice(0, 0, N);
        auto chunk_table_slice = input.aux->device_chunk_table.slice(0, 0, N);
        chunks_slice.slice(1, 0, 2).copy_(chunk_table_slice);
        chunks_slice.select(1, 2).copy_(chunk_table_slice.select(1, 0));
        chunks.index({at::indexing::Slice(0, N), 3}) = 0;
        chunk_results = at::empty({N_, 8}, tensor_options_int32);
    } else {
        chunks = at::empty({N, 4}, tensor_options_int32);
        chunks.index({at::indexing::Slice(), 0}) =
                at::arange(0, int(T * N), int(T), tensor_options_int32);
        chunks.index({at::indexing::Slice(), 2}) =
                at::arange(0, int(T * N), int(T), tensor_options_int32);
        chunks.index({at::indexing::Slice(), 1}) = int(T);
        chunks.index({at::indexing::Slice(), 3}) = 0;
        chunk_results = at::empty({N, 8}, tensor_options_int32);
    }

    at::Tensor aux;
    at::Tensor path;
    at::Tensor moves_sequence_qstring;
    if (input.aux) {
        if (input.aux->is_lstm_or_flstm_model()) {
            // lstm VCS
            const std::int32_t T_ = input.aux->NT_out_max();
            const std::int32_t Ts_ = std::max<std::int32_t>(T_ + (input.aux->N() * 4), T + N);
            aux = at::empty(Ts_ * (C + 4 * options.beam_width), tensor_options_int8);
            path = at::zeros(Ts_, tensor_options_int32);
            moves_sequence_qstring = at::zeros({3, T_}, tensor_options_int8);
        } else {
            // Tx VCS
            aux = at::empty(
                    input.aux->total_num_granularity() * (T + 1) * (C + 4 * options.beam_width),
                    tensor_options_int8);
            path = at::zeros(input.aux->total_num_granularity() * (T + 1), tensor_options_int32);
            moves_sequence_qstring =
                    at::zeros({3, input.aux->total_num_granularity() * T}, tensor_options_int8);
        }
    } else {
        aux = at::empty(N * (T + 1) * (C + 4 * options.beam_width), tensor_options_int8);
        path = at::zeros(N * (T + 1), tensor_options_int32);
        moves_sequence_qstring = at::zeros({3, N * T}, tensor_options_int8);
    }

    auto moves = moves_sequence_qstring[0];
    auto sequence = moves_sequence_qstring[1];
    auto qstring = moves_sequence_qstring[2];

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    KoiTypeId scores_type_id = (scores.dtype() == at::kChar) ? KOI_I8 : KOI_F16;
    if (utils::get_dev_opt("fused_decode", true)) {
        utils::ScopedProfileRange spr{"fused_beam_search", 2};
        const int flags = utils::get_dev_opt("fused_decode_flags", (C == 1024) ? 1 : 0);
        dorado::utils::handle_cuda_result(koi_fused_beam_search(
                stream, chunks.data_ptr(), chunk_results.data_ptr(), N, scores.data_ptr(),
                m_score_range, C, scores_type_id, aux.data_ptr(), int(options.beam_width),
                options.beam_cut, options.blank_score, flags));
    } else {
        {
            utils::ScopedProfileRange spr{"back_guides", 2};
            dorado::utils::handle_cuda_result(koi_compute_back_guides(
                    stream, chunks.data_ptr(), N, scores.data_ptr(), m_score_range, C,
                    scores_type_id, aux.data_ptr(), options.blank_score));
        }
        {
            utils::ScopedProfileRange spr{"beam_search", 2};
            dorado::utils::handle_cuda_result(koi_beam_search(
                    stream, chunks.data_ptr(), chunk_results.data_ptr(), N, scores.data_ptr(),
                    m_score_range, C, scores_type_id, aux.data_ptr(), int(options.beam_width),
                    options.beam_cut, options.blank_score));
        }
        {
            utils::ScopedProfileRange spr{"compute_posts", 2};
            dorado::utils::handle_cuda_result(koi_compute_posts(
                    stream, chunks.data_ptr(), N, scores.data_ptr(), m_score_range, C,
                    scores_type_id, aux.data_ptr(), options.blank_score));
        }
    }
    {
        utils::ScopedProfileRange spr{"beam_decode", 2};
        dorado::utils::handle_cuda_result(koi_beam_decode(
                stream, chunks.data_ptr(), chunk_results.data_ptr(), N, scores.data_ptr(), C,
                scores_type_id, aux.data_ptr(), int(options.beam_width), path.data_ptr(),
                moves.data_ptr(), sequence.data_ptr(), qstring.data_ptr(), options.q_scale,
                options.q_shift, options.move_pad));
    }
    auto data = input.aux ? moves_sequence_qstring : moves_sequence_qstring.reshape({3, N, -1});
    return {std::move(data), {}, input.num_chunks, input.options, input.aux};
}

std::vector<DecodedChunk> CUDADecoder::decode(const DecoderResult &data) const {
    nvtx3::scoped_range loop{"cpu_decode"};

    const at::Tensor &moves_sequence_qstring_cpu = data.data;
    assert(moves_sequence_qstring_cpu.device() == at::kCPU);
    auto moves_cpu = moves_sequence_qstring_cpu[0];
    auto sequence_cpu = moves_sequence_qstring_cpu[1];
    auto qstring_cpu = moves_sequence_qstring_cpu[2];

    std::vector<DecodedChunk> called_chunks;

    // N = total_num_varlen_chunks for Tx VCS too
    if (data.aux) {
        const std::span<const std::int32_t> chunk_sizes(data.aux->chunk_sizes());
        const std::int32_t N = std::ssize(chunk_sizes);
        const auto *const moves_ptr = static_cast<const uint8_t *>(moves_cpu.const_data_ptr());
        const auto *const seq_ptr = static_cast<const char *>(sequence_cpu.const_data_ptr());
        const auto *const qstr_ptr = static_cast<const char *>(qstring_cpu.const_data_ptr());

        std::ptrdiff_t offset = 0;
        called_chunks.reserve(N);
        for (std::int32_t idx = 0; idx < N; ++idx) {
            const std::int32_t T = chunk_sizes[idx];

            std::vector<uint8_t> mov(moves_ptr + offset, moves_ptr + offset + T);
            const auto num_bases = std::reduce(mov.cbegin(), mov.cend(), std::size_t{0});
            std::string seq(seq_ptr + offset, seq_ptr + offset + num_bases);
            std::string qstr(qstr_ptr + offset, qstr_ptr + offset + num_bases);

            called_chunks.emplace_back(
                    DecodedChunk{std::move(seq), std::move(qstr), std::move(mov), {}, {}});

            offset += T;
        }

    } else {
        const int N = static_cast<int>(moves_cpu.size(0));
        const int T = static_cast<int>(moves_cpu.size(1));

        called_chunks.reserve(N);
        for (int idx = 0; idx < N; idx++) {
            const auto *const moves_ptr =
                    static_cast<const uint8_t *>(moves_cpu[idx].const_data_ptr());
            const auto *const seq_ptr =
                    static_cast<const char *>(sequence_cpu[idx].const_data_ptr());
            const auto *const qstr_ptr =
                    static_cast<const char *>(qstring_cpu[idx].const_data_ptr());

            std::vector<uint8_t> mov(moves_ptr, moves_ptr + T);
            const auto num_bases = std::reduce(mov.cbegin(), mov.cend(), std::size_t{0});
            std::string seq(seq_ptr, seq_ptr + num_bases);
            std::string qstr(qstr_ptr, qstr_ptr + num_bases);

            std::vector<at::Tensor> modbase_data;
            modbase_data.reserve(data.modbase_data.size());
            for (const at::Tensor &head : data.modbase_data) {
                modbase_data.emplace_back(head[idx]);
            }

            auto [modbase_probs, modbase_motif_hits] = decode_modbase(modbase_data, seq, mov);

            called_chunks.emplace_back(std::move(seq), std::move(qstr), std::move(mov),
                                       std::move(modbase_probs), std::move(modbase_motif_hits));
        }
    }

    return called_chunks;
}

std::pair<std::vector<uint8_t>, std::vector<bool>> CUDADecoder::decode_modbase(
        const std::span<const at::Tensor> data,
        const std::string_view sequence,
        const std::span<const uint8_t> moves) const {
    if (m_modbase_decoders.empty()) {
        return {};
    }

    assert(data.size() == m_modbase_decoders.size());

    std::vector<int32_t> sequence_to_signal;
    sequence_to_signal.reserve(sequence.size());
    for (int32_t i = 0; i < std::ssize(moves); ++i) {
        if (moves[i] == 1) {
            sequence_to_signal.emplace_back(i);
        }
    }
    assert(sequence_to_signal.size() == sequence.size());

    std::vector<uint8_t> modbase_probabilities(sequence.size() * m_modbase_num_states, 0);
    for (int32_t i = 0; i < std::ssize(sequence); ++i) {
        const int8_t base_id{utils::BaseInfo::BASE_IDS[sequence[i]]};
        assert((base_id >= 0) && (base_id <= 3));
        modbase_probabilities[(i * m_modbase_num_states) + m_modbase_base_offsets[base_id]] = 255U;
    }

    std::vector<bool> modbase_motif_hits(sequence.size(), false);

    for (int32_t i = 0; i < std::ssize(data); ++i) {
        assert(data[i].device() == at::kCPU);
        assert(data[i].dim() == 1);

        const ModBaseDecoder &decoder{m_modbase_decoders[i]};
        assert(data[i].size(0) == (decoder.num_states * std::ssize(moves)));

        const std::span<const at::Half> scores(data[i].data_ptr<at::Half>(), data[i].size(0));

        const size_t base_offset = m_modbase_base_offsets[decoder.base_id];

        const std::vector<size_t> sequence_hits{decoder.matcher.get_motif_hits(sequence)};
        for (const int32_t sequence_hit : sequence_hits) {
            const int32_t signal_hit = sequence_to_signal[sequence_hit];
            if (signal_hit < decoder.context_num_samples_before) {
                continue;
            }
            if (signal_hit > (std::ssize(moves) - decoder.context_num_samples_after)) {
                break;
            }
            modbase_motif_hits[sequence_hit] = true;
            for (int32_t mod_offset = 0; mod_offset < decoder.num_states; ++mod_offset) {
                const uint8_t score = std::min(
                        255.0f,
                        std::floor(
                                static_cast<float>(
                                        scores[((signal_hit * decoder.num_states)) + mod_offset]) *
                                256));
                modbase_probabilities[(sequence_hit * m_modbase_num_states) + base_offset +
                                      mod_offset] = score;
            }
        }
    }

    return {modbase_probabilities, modbase_motif_hits};
}

}  // namespace dorado::basecall::decode
