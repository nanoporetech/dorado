#include "CPUDecoder.h"

#include "beam_search.h"

#include <math.h>
#include <torch/torch.h>

#include <vector>

at::Tensor scan(at::Tensor Ms, at::Tensor idx, at::Tensor v0) {
    int T = Ms.size(0);
    int N = Ms.size(1);
    int C = Ms.size(2);

    torch::Tensor alpha = Ms.new_full({T + 1, N, C}, -1E38);
    alpha[0] = v0;

    for (int t = 0; t < T; t++) {
        auto x = torch::add(Ms[t], alpha.index({t, torch::indexing::Slice(), idx}));
        alpha[t + 1] = torch::logsumexp(x, -1);
    }

    return alpha;
}

torch::Tensor CPUDecoder::forward_scores(at::Tensor scores) {
    int T = scores.size(0);  // Signal len
    int N = scores.size(1);  // Num batches
    int C = scores.size(2);

    int n_base = 4;
    int num_transitions = 5;

    int state_len = log(C / num_transitions) / log(n_base);

    torch::Tensor Ms = scores.reshape({T, N, -1, num_transitions});

    int y = pow(n_base, state_len);

    auto v0 = Ms.new_full({{N, y}}, 0.0);

    auto t1 = torch::arange(y).index({torch::indexing::Slice(), torch::indexing::None});
    auto t2 = torch::arange(y).repeat_interleave(n_base).reshape({n_base, -1});
    t2 = t2.t().contiguous();

    auto idx = torch::cat({t1, t2}, 1).to(torch::kInt32);

    auto result = scan(Ms, idx.to(torch::kInt64), v0);
    return result;
}

torch::Tensor CPUDecoder::backward_scores(torch::Tensor scores) {
    int N = scores.size(1);  // Num batches
    int C = scores.size(2);  // Num batches

    int n_base = 4;
    int num_transitions = 5;

    int state_len = log(C / num_transitions) / log(n_base);

    int y = pow(n_base, state_len);

    torch::Tensor vT = scores.new_full({N, y}, 0.0);

    auto t1 = torch::arange(y).index({torch::indexing::Slice(), torch::indexing::None});
    auto t2 = torch::arange(y).repeat_interleave(n_base).reshape({n_base, -1}).t().contiguous();
    auto idx = torch::cat({t1, t2}, 1).to(torch::kInt32);

    auto idx_sizes = idx.sizes();
    auto idx_T = idx.flatten().argsort().reshape(idx_sizes);

    auto Ms_T = scores.index({torch::indexing::Slice(), torch::indexing::Slice(), idx_T});

    idx_T = torch::div(idx_T, n_base + 1, "floor");

    auto result = scan(Ms_T.flip(0), idx_T.to(torch::kInt64), vT).flip(0);
    return result;
}

std::vector<DecodedChunk> CPUDecoder::beam_search(torch::Tensor scores,
                                                  int num_chunks,
                                                  DecoderOptions options) {
    scores = scores.to("cpu");
    int num_threads = std::min(num_chunks, 4);
    int chunks_per_thread = num_chunks / num_threads;
    int num_threads_with_one_more_chunk = num_chunks % num_threads;

    std::vector<DecodedChunk> chunk_results(num_chunks);

    std::vector<std::unique_ptr<std::thread>> threads;
    threads.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(new std::thread(
                [&](int i) {
                    int t_first_chunk =
                            i * chunks_per_thread + std::min(i, num_threads_with_one_more_chunk);
                    int t_num_chunks = chunks_per_thread + int(i < num_threads_with_one_more_chunk);

                    using Slice = torch::indexing::Slice;
                    auto t_scores = scores.index(
                            {Slice(), Slice(t_first_chunk, t_first_chunk + t_num_chunks)});

                    torch::Tensor fwd = forward_scores(t_scores);
                    torch::Tensor bwd = backward_scores(t_scores);

                    torch::Tensor posts = torch::softmax(fwd + bwd, -1);

                    t_scores = t_scores.transpose(0, 1);
                    bwd = bwd.transpose(0, 1).contiguous();
                    posts = posts.transpose(0, 1).contiguous();

                    for (int i = 0; i < t_num_chunks; i++) {
                        auto decode_result =
                                beam_search_decode(t_scores[i], bwd[i], posts[i], options.beam_cut,
                                                   options.blank_score, options.q_shift,
                                                   options.q_scale, options.temperature);
                        chunk_results[t_first_chunk + i] = DecodedChunk{
                                std::get<0>(decode_result),
                                std::get<1>(decode_result),
                                std::get<2>(decode_result),
                        };
                    }
                },
                i));
    }

    for (auto& thread : threads) {
        thread->join();
    }

    return chunk_results;
}
