#include "CUDADecoder.h"

#include "torch_utils/cuda_utils.h"
#include "torch_utils/gpu_profiling.h"

#include <c10/cuda/CUDAGuard.h>
#include <nvtx3/nvtx3.hpp>

extern "C" {
#include "koi.h"
}

namespace dorado::basecall::decode {

DecodeData CUDADecoder::beam_search_part_1(DecodeData data) const {
    auto scores = data.data;
    auto &options = data.options;

    c10::cuda::CUDAGuard device_guard(scores.device());
    utils::ScopedProfileRange loop{"gpu_decode", 1};
    long int N = (long int)(scores.sizes()[0]);
    long int T = (long int)(scores.sizes()[1]);
    long int C = (long int)(scores.sizes()[2]);

    auto tensor_options_int32 =
            at::TensorOptions().dtype(at::kInt).device(scores.device()).requires_grad(false);

    auto tensor_options_int8 =
            at::TensorOptions().dtype(at::kChar).device(scores.device()).requires_grad(false);

    auto chunks = at::empty({N, 4}, tensor_options_int32);
    chunks.index({at::indexing::Slice(), 0}) =
            at::arange(0, int(T * N), int(T), tensor_options_int32);
    chunks.index({at::indexing::Slice(), 2}) =
            at::arange(0, int(T * N), int(T), tensor_options_int32);
    chunks.index({at::indexing::Slice(), 1}) = int(T);
    chunks.index({at::indexing::Slice(), 3}) = 0;

    auto chunk_results = at::empty({N, 8}, tensor_options_int32);

    chunk_results = chunk_results.contiguous();

    auto aux = at::empty(N * (T + 1) * (C + 4 * options.beam_width), tensor_options_int8);
    auto path = at::zeros(N * (T + 1), tensor_options_int32);

    auto moves_sequence_qstring = at::zeros({3, N * T}, tensor_options_int8);
    auto moves = moves_sequence_qstring[0];
    auto sequence = moves_sequence_qstring[1];
    auto qstring = moves_sequence_qstring[2];

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    {
        utils::ScopedProfileRange spr{"back_guides", 2};
        dorado::utils::handle_cuda_result(host_back_guide_step(
                stream, chunks.data_ptr(), chunk_results.data_ptr(), N, scores.data_ptr(),
                m_score_clamp_val, C, aux.data_ptr(), path.data_ptr(), moves.data_ptr(), NULL,
                sequence.data_ptr(), qstring.data_ptr(), options.q_scale, options.q_shift,
                int(options.beam_width), options.beam_cut, options.blank_score));
    }
    {
        utils::ScopedProfileRange spr{"beam_search", 2};
        dorado::utils::handle_cuda_result(host_beam_search_step(
                stream, chunks.data_ptr(), chunk_results.data_ptr(), N, scores.data_ptr(),
                m_score_clamp_val, C, aux.data_ptr(), path.data_ptr(), moves.data_ptr(), NULL,
                sequence.data_ptr(), qstring.data_ptr(), options.q_scale, options.q_shift,
                int(options.beam_width), options.beam_cut, options.blank_score));
    }
    {
        utils::ScopedProfileRange spr{"compute_posts", 2};
        dorado::utils::handle_cuda_result(host_compute_posts_step(
                stream, chunks.data_ptr(), chunk_results.data_ptr(), N, scores.data_ptr(),
                m_score_clamp_val, C, aux.data_ptr(), path.data_ptr(), moves.data_ptr(), NULL,
                sequence.data_ptr(), qstring.data_ptr(), options.q_scale, options.q_shift,
                int(options.beam_width), options.beam_cut, options.blank_score));
    }
    {
        utils::ScopedProfileRange spr{"decode", 2};
        dorado::utils::handle_cuda_result(host_run_decode(
                stream, chunks.data_ptr(), chunk_results.data_ptr(), N, scores.data_ptr(),
                m_score_clamp_val, C, aux.data_ptr(), path.data_ptr(), moves.data_ptr(), NULL,
                sequence.data_ptr(), qstring.data_ptr(), options.q_scale, options.q_shift,
                int(options.beam_width), options.beam_cut, options.blank_score, options.move_pad));
    }

    data.data = moves_sequence_qstring.reshape({3, N, -1});
    return data;
}

std::vector<DecodedChunk> CUDADecoder::beam_search_part_2(DecodeData data) const {
    auto moves_sequence_qstring_cpu = data.data;
    nvtx3::scoped_range loop{"cpu_decode"};
    assert(moves_sequence_qstring_cpu.device() == at::kCPU);
    auto moves_cpu = moves_sequence_qstring_cpu[0];
    auto sequence_cpu = moves_sequence_qstring_cpu[1];
    auto qstring_cpu = moves_sequence_qstring_cpu[2];
    int N = int(moves_cpu.size(0));
    int T = int(moves_cpu.size(1));

    std::vector<DecodedChunk> called_chunks;

    for (int idx = 0; idx < N; idx++) {
        std::vector<uint8_t> mov((uint8_t *)moves_cpu[idx].data_ptr(),
                                 (uint8_t *)moves_cpu[idx].data_ptr() + T);
        auto num_bases = moves_cpu[idx].sum().item<int>();
        std::string seq((char *)sequence_cpu[idx].data_ptr(),
                        (char *)sequence_cpu[idx].data_ptr() + num_bases);
        std::string qstr((char *)qstring_cpu[idx].data_ptr(),
                         (char *)qstring_cpu[idx].data_ptr() + num_bases);

        called_chunks.emplace_back(DecodedChunk{std::move(seq), std::move(qstr), std::move(mov)});
    }

    return called_chunks;
}

}  // namespace dorado::basecall::decode
