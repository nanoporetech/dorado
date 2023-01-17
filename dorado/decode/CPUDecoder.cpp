#include "CPUDecoder.h"

#include "beam_search.h"

#include <math.h>
#include <spdlog/spdlog.h>
#include <torch/torch.h>

#include <vector>

namespace {

at::Tensor scan(const torch::Tensor& Ms,
                const float fixed_stay_score,
                const torch::Tensor& idx,
                const torch::Tensor& v0) {
    const int T = Ms.size(0);
    const int N = Ms.size(1);
    const int C = Ms.size(2);

    torch::Tensor alpha = Ms.new_full({T + 1, N, C}, -1E38);
    alpha[0] = v0;

    for (int t = 0; t < T; t++) {
        auto scored_steps = torch::add(alpha.index({t, torch::indexing::Slice(), idx}), Ms[t]);
        auto scored_stay = torch::add(alpha.index({t, torch::indexing::Slice()}), fixed_stay_score)
                                   .unsqueeze(-1);
        auto scored_transitions = torch::cat({scored_stay, scored_steps}, -1);

        alpha[t + 1] = torch::logsumexp(scored_transitions, -1);
    }

    return alpha;
}

torch::Tensor forward_scores(const torch::Tensor& scores, const float fixed_stay_score) {
    const int T = scores.size(0);  // Signal len
    const int N = scores.size(1);  // Num batches
    const int C = scores.size(2);  // 4^state_len * 4 = 4^(state_len + 1)

    const int n_base = 4;
    const int state_len = std::log(C) / std::log(n_base) - 1;

    // Transition scores reshaped so that the 4 scores for each predecessor state are arranged along the
    // innermost dimension.
    const torch::Tensor Ms = scores.reshape({T, N, -1, n_base});

    // Number of states per timestep.
    const int num_states = pow(n_base, state_len);

    // Guide values at first timestep.
    const auto v0 = Ms.new_full({{N, num_states}}, 0.0f);

    // For each state, the indices of the 4 states that could precede it via a step transition.
    const auto idx = torch::arange(num_states)
                             .repeat_interleave(n_base)
                             .reshape({n_base, -1})
                             .t()
                             .contiguous();

    return scan(Ms, fixed_stay_score, idx, v0);
}

torch::Tensor backward_scores(const torch::Tensor& scores, const float fixed_stay_score) {
    const int N = scores.size(1);  // Num batches
    const int C = scores.size(2);  // 4^state_len * 4 = 4^(state_len + 1)

    const int n_base = 4;

    const int state_len = std::log(C) / std::log(n_base) - 1;

    // Number of states per timestep.
    const int num_states = pow(n_base, state_len);

    // Guide values at last timestep.
    const torch::Tensor vT = scores.new_full({N, num_states}, 0.0f);

    const auto idx = torch::arange(num_states)
                             .repeat_interleave(n_base)
                             .reshape({n_base, -1})
                             .t()
                             .contiguous();
    auto idx_T = idx.flatten().argsort().reshape(idx.sizes());

    const auto Ms_T = scores.index({torch::indexing::Slice(), torch::indexing::Slice(), idx_T});

    // For each state, the indices of the 4 states that could succeed it via a step transition.
    idx_T = torch::bitwise_right_shift(idx_T, 2);

    return scan(Ms_T.flip(0), fixed_stay_score, idx_T.to(torch::kInt64), vT).flip(0);
}

}  // namespace

namespace dorado {

std::vector<DecodedChunk> CPUDecoder::beam_search(const torch::Tensor& scores,
                                                  const int num_chunks,
                                                  const DecoderOptions& options) {
    const auto scores_cpu = scores.to(torch::kCPU);
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
                    auto t_scores = scores_cpu.index(
                            {Slice(), Slice(t_first_chunk, t_first_chunk + t_num_chunks)});

                    torch::Tensor fwd = forward_scores(t_scores, options.blank_score);
                    torch::Tensor bwd = backward_scores(t_scores, options.blank_score);

                    torch::Tensor posts = torch::softmax(fwd + bwd, -1);

                    t_scores = t_scores.transpose(0, 1);
                    bwd = bwd.transpose(0, 1).contiguous();
                    posts = posts.transpose(0, 1).contiguous();

                    for (int i = 0; i < t_num_chunks; i++) {
                        auto decode_result = beam_search_decode(
                                t_scores[i], bwd[i], posts[i], options.beam_width, options.beam_cut,
                                options.blank_score, options.q_shift, options.q_scale,
                                options.temperature, 1.0f);
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

}  // namespace dorado
