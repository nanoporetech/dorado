#include "GPUDecoder.h"

#include "Decoder.h"

#include <torch/torch.h>

#ifndef __APPLE__
extern "C" {
#include "koi.h"
}
#include <cuda_runtime.h>
#endif

std::vector<DecodedChunk> GPUDecoder::beam_search(torch::Tensor scores,
                                                  int num_chunks,
                                                  DecoderOptions options) {
    scores = scores.transpose(1, 0).contiguous();

    long int N = scores.sizes()[0];
    long int T = scores.sizes()[1];
    long int C = scores.sizes()[2];

    auto tensor_options_int32 = torch::TensorOptions()
                                        .dtype(torch::kInt32)
                                        .device(scores.device())
                                        .requires_grad(false);

    auto tensor_options_int8 =
            torch::TensorOptions().dtype(torch::kInt8).device(scores.device()).requires_grad(false);

    if (!initialized) {
        chunks = torch::empty({N, 4}, tensor_options_int32);
        chunks.index({torch::indexing::Slice(), 0}) = torch::arange(0, int(T * N), int(T));
        chunks.index({torch::indexing::Slice(), 2}) = torch::arange(0, int(T * N), int(T));
        chunks.index({torch::indexing::Slice(), 1}) = int(T);
        chunks.index({torch::indexing::Slice(), 3}) = 0;

        chunk_results = torch::empty({N, 8}, tensor_options_int32);

        chunk_results = chunk_results.contiguous();

        aux = torch::empty(N * (T + 1) * (C + 4 * options.beam_width), tensor_options_int8);
        path = torch::zeros(N * (T + 1), tensor_options_int32);

        moves = torch::zeros(N * T, tensor_options_int8);
        sequence = torch::zeros(N * T, tensor_options_int8);
        qstring = torch::zeros(N * T, tensor_options_int8);

        initialized = true;
    }

    moves.index({torch::indexing::Slice()}) = 0.0;
    sequence.index({torch::indexing::Slice()}) = 0.0;
    qstring.index({torch::indexing::Slice()}) = 0.0;

#ifndef __APPLE__
    int cuda_device_id = get_cuda_device_id_from_device(scores.device());
    if (cudaSetDevice(cuda_device_id) != cudaSuccess) {
        throw std::runtime_error("Unable to set cuda device!");
    }
    host_back_guide_step(chunks.data_ptr(), chunk_results.data_ptr(), N, scores.data_ptr(), C,
                         aux.data_ptr(), path.data_ptr(), moves.data_ptr(), NULL,
                         sequence.data_ptr(), qstring.data_ptr(), options.q_scale, options.q_shift,
                         options.beam_width, options.beam_cut, options.blank_score);

    host_beam_search_step(chunks.data_ptr(), chunk_results.data_ptr(), N, scores.data_ptr(), C,
                          aux.data_ptr(), path.data_ptr(), moves.data_ptr(), NULL,
                          sequence.data_ptr(), qstring.data_ptr(), options.q_scale, options.q_shift,
                          options.beam_width, options.beam_cut, options.blank_score);

    host_compute_posts_step(chunks.data_ptr(), chunk_results.data_ptr(), N, scores.data_ptr(), C,
                            aux.data_ptr(), path.data_ptr(), moves.data_ptr(), NULL,
                            sequence.data_ptr(), qstring.data_ptr(), options.q_scale,
                            options.q_shift, options.beam_width, options.beam_cut,
                            options.blank_score);

    host_run_decode(chunks.data_ptr(), chunk_results.data_ptr(), N, scores.data_ptr(), C,
                    aux.data_ptr(), path.data_ptr(), moves.data_ptr(), NULL, sequence.data_ptr(),
                    qstring.data_ptr(), options.q_scale, options.q_shift, options.beam_width,
                    options.beam_cut, options.blank_score, options.move_pad);

#endif

    auto sequence_cpu = sequence.reshape({N, -1}).to(torch::kCPU);
    auto qstring_cpu = qstring.reshape({N, -1}).to(torch::kCPU);
    auto moves_cpu = moves.reshape({N, -1}).to(torch::kCPU);

    std::vector<DecodedChunk> called_chunks;

    for (int idx = 0; idx < N; idx++) {
        std::vector<uint8_t> mov((uint8_t*)moves_cpu[idx].data_ptr(),
                                 (uint8_t*)moves_cpu[idx].data_ptr() + T);
        auto num_bases = moves_cpu[idx].sum().item<int>();
        std::string seq((char*)sequence_cpu[idx].data_ptr(),
                        (char*)sequence_cpu[idx].data_ptr() + num_bases);
        std::string qstr((char*)qstring_cpu[idx].data_ptr(),
                         (char*)qstring_cpu[idx].data_ptr() + num_bases);

        called_chunks.emplace_back(DecodedChunk{std::move(seq), std::move(qstr), std::move(mov)});
    }

    return called_chunks;
}

int GPUDecoder::get_cuda_device_id_from_device(const c10::Device& device) {
    if (!device.is_cuda() || !device.has_index()) {
        std::stringstream ss;
        ss << "Unable to extract CUDA device ID from device " << device;
        throw std::runtime_error(ss.str());
    }

    return device.index();
}
