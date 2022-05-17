#include <math.h>
#include <vector>
#include <torch/torch.h>
#include "MTLDecoder.h"
#include "beam_search.h"

using Slice = torch::indexing::Slice;

MTLDecoder::MTLDecoder() {
    device = get_mtl_device();
    command_queue = device->newCommandQueue();
    scan_cps = make_cps(device, "scan");
}

std::vector<DecodedChunk> MTLDecoder::beam_search(torch::Tensor scores, int num_chunks, DecoderOptions options) {
    constexpr int n_base = 4;
    constexpr int num_transitions = 5;

    int T = scores.size(0);
    int N = scores.size(1);
    int C = scores.size(2);
    int Cs = C / num_transitions;

    if (scan_idx[0][0].size(0) != C) {
        int state_len = log(Cs) / log(n_base);
        int y = pow(n_base,state_len);

        scan_idx[0][0] = torch::arange(C, torch::kInt32).contiguous();
        auto t1 = torch::arange(y).index({torch::indexing::Slice(), torch::indexing::None});
        auto t2 = torch::arange(y).repeat_interleave(n_base).reshape({n_base, -1}).t();
        scan_idx[0][1] = torch::cat({t1,t2}, 1).to(torch::kInt32).contiguous();

        auto idx_sizes = scan_idx[0][1].sizes();
        scan_idx[1][0] = scan_idx[0][1].flatten().argsort().reshape(idx_sizes).to(torch::kInt32).contiguous();
        scan_idx[1][1] = torch::div(scan_idx[1][0], num_transitions, "floor");
    }

    auto fwd = torch::empty({N, T+1, Cs});
    auto bwd = torch::empty({N, T+1, Cs});
    int32_t scan_args_[] = {T, N, Cs, 1}; // T, N, C, dir
    auto args_fwd = create_buffer(device, scan_args_, 4);
    scan_args_[3] = -1;
    auto args_bwd = create_buffer(device, scan_args_, 4);

    auto command_buffer = command_queue->commandBuffer();
    launch_kernel_no_wait(scan_cps, command_buffer, {args_fwd, mtl_for_tensor(scores), mtl_for_tensor(fwd), mtl_for_tensor(scan_idx[0][0]), mtl_for_tensor(scan_idx[0][1])}, N, Cs);
    launch_kernel_no_wait(scan_cps, command_buffer, {args_bwd, mtl_for_tensor(scores), mtl_for_tensor(bwd), mtl_for_tensor(scan_idx[1][0]), mtl_for_tensor(scan_idx[1][1])}, N, Cs);
    command_buffer->commit();
    command_buffer->waitUntilCompleted();

    constexpr int num_threads = 4;
    int chunks_per_thread = (num_chunks + num_threads - 1) / num_threads;

    std::vector<DecodedChunk> chunk_results(num_chunks);

    std::vector<std::unique_ptr<std::thread>> threads;
    threads.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(
            new std::thread(
                [&] (int i) {
                    int t_first_chunk = i * chunks_per_thread;
                    int t_num_chunks = std::min(num_chunks - t_first_chunk, chunks_per_thread);

                    Slice t_slice = Slice(t_first_chunk, t_first_chunk + t_num_chunks);
                    auto t_scores = scores.index({Slice(), t_slice, Slice()}).transpose(0, 1).contiguous();;
                    auto t_fwd = fwd.index({t_slice});
                    auto t_bwd = bwd.index({t_slice});

                    auto posts = torch::softmax(t_fwd + t_bwd, -1);

                    for (int i = 0; i < t_num_chunks; i++) {
                        auto decode_result = beam_search_decode(t_scores[i],
                                                                t_bwd[i],
                                                                posts[i],
                                                                options.beam_cut,
                                                                options.blank_score,
                                                                options.q_shift,
                                                                options.q_scale,
                                                                options.temperature);
                        chunk_results[t_first_chunk + i] = DecodedChunk{
                                std::get<0>(decode_result),
                                std::get<1>(decode_result),
                                std::get<2>(decode_result),
                        };
                    }
                }
            , i));
    }

    for (auto& thread : threads) {
        thread->join();
    }

    return chunk_results;
}

