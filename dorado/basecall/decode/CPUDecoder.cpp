#include "CPUDecoder.h"

#include "beam_search.h"

#include <ATen/Functions.h>
#include <ATen/TensorIndexing.h>
#include <ATen/TensorOperators.h>
#include <math.h>
#include <spdlog/spdlog.h>

#include <vector>

namespace {

// Operates in TNC
at::Tensor scan(const at::Tensor& Ms,
                const float fixed_stay_score,
                const at::Tensor& idx,
                const at::Tensor& v0) {
    const int T = int(Ms.size(0));
    const int N = int(Ms.size(1));
    const int C = int(Ms.size(2));

    at::Tensor alpha = Ms.new_full({T + 1, N, C}, -1E38);
    alpha[0] = v0;

    for (int t = 0; t < T; t++) {
        auto scored_steps = at::add(alpha.index({t, at::indexing::Slice(), idx}), Ms[t]);
        auto scored_stay =
                at::add(alpha.index({t, at::indexing::Slice()}), fixed_stay_score).unsqueeze(-1);
        auto scored_transitions = at::cat({scored_stay, scored_steps}, -1);

        alpha[t + 1] = at::logsumexp(scored_transitions, -1);
    }

    return alpha;
}
}  // namespace

namespace dorado::basecall::decode::inner {

at::Tensor forward_scores(const at::Tensor& scores_TNC, const float fixed_stay_score) {
    const int T = int(scores_TNC.size(0));  // Signal len
    const int N = int(scores_TNC.size(1));  // Num batches
    const int C = int(scores_TNC.size(2));  // 4^state_len * 4 = 4^(state_len + 1)

    const int n_base = 4;
    const int state_len = int(std::log(C) / std::log(n_base) - 1);

    // Transition scores reshaped so that the 4 scores for each predecessor state are arranged along the
    // innermost dimension.
    const at::Tensor Ms = scores_TNC.reshape({T, N, -1, n_base});

    // Number of states per timestep.
    const int num_states = int(pow(n_base, state_len));

    // Guide values at first timestep.
    const auto v0 = Ms.new_full({{N, num_states}}, 0.0f);

    // For each state, the indices of the 4 states that could precede it via a step transition.
    const auto idx =
            at::arange(num_states).repeat_interleave(n_base).reshape({n_base, -1}).t().contiguous();

    return scan(Ms, fixed_stay_score, idx, v0);
}

at::Tensor backward_scores(const at::Tensor& scores_TNC, const float fixed_stay_score) {
    const int N = int(scores_TNC.size(1));  // Num batches
    const int C = int(scores_TNC.size(2));  // 4^state_len * 4 = 4^(state_len + 1)

    const int n_base = 4;

    const int state_len = int(std::log(C) / std::log(n_base) - 1);

    // Number of states per timestep.
    const int num_states = int(pow(n_base, state_len));

    // Guide values at last timestep.
    const at::Tensor vT = scores_TNC.new_full({N, num_states}, 0.0f);

    const auto idx =
            at::arange(num_states).repeat_interleave(n_base).reshape({n_base, -1}).t().contiguous();
    auto idx_T = idx.flatten().argsort().reshape(idx.sizes());

    const auto Ms_T = scores_TNC.index({at::indexing::Slice(), at::indexing::Slice(), idx_T});

    // For each state, the indices of the 4 states that could succeed it via a step transition.
    idx_T = at::bitwise_right_shift(idx_T, 2);

    return scan(Ms_T.flip(0), fixed_stay_score, idx_T.to(at::kLong), vT).flip(0);
}

}  // namespace dorado::basecall::decode::inner

namespace dorado::basecall::decode {

DecodeData CPUDecoder::beam_search_part_1(DecodeData data) const { return data; }

std::vector<DecodedChunk> CPUDecoder::beam_search_part_2(DecodeData data) const {
    // Expects data.data(TNC)
    const auto scores_cpu = data.data.to(at::kCPU);
    const auto num_chunks = data.num_chunks;
    const auto& options = data.options;
    int num_threads = std::min(num_chunks, 4);
    int chunks_per_thread = num_chunks / num_threads;
    int num_threads_with_one_more_chunk = num_chunks % num_threads;

    std::vector<DecodedChunk> chunk_results(num_chunks);

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            at::InferenceMode inference_mode_guard;

            int t_first_chunk =
                    i * chunks_per_thread + std::min(i, num_threads_with_one_more_chunk);
            int t_num_chunks = chunks_per_thread + int(i < num_threads_with_one_more_chunk);

            // Slice TNC -> TnC
            using Slice = at::indexing::Slice;
            auto t_scores =
                    scores_cpu.index({Slice(), Slice(t_first_chunk, t_first_chunk + t_num_chunks)});

            at::Tensor fwd = inner::forward_scores(t_scores, options.blank_score);
            at::Tensor bwd = inner::backward_scores(t_scores, options.blank_score);

            at::Tensor posts = at::softmax(fwd + bwd, -1);

            // Transpose TnC to nTC
            t_scores = t_scores.transpose(0, 1);
            bwd = bwd.transpose(0, 1).contiguous();
            posts = posts.transpose(0, 1).contiguous();

            // Iter over n in nTC, passing TC tensors to beam_search_decode
            for (int chunk_idx = 0; chunk_idx < t_num_chunks; chunk_idx++) {
                auto decode_result = beam_search_decode(t_scores[chunk_idx], bwd[chunk_idx],
                                                        posts[chunk_idx], options.beam_width,
                                                        options.beam_cut, options.blank_score,
                                                        options.q_shift, options.q_scale, 1.0f);
                chunk_results[t_first_chunk + chunk_idx] = DecodedChunk{
                        std::get<0>(decode_result),
                        std::get<1>(decode_result),
                        std::get<2>(decode_result),
                };
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return chunk_results;
}

}  // namespace dorado::basecall::decode
