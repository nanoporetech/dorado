#pragma once

#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

std::tuple<std::string, std::string, std::vector<uint8_t>> beam_search_decode(
        const torch::Tensor& scores_t,
        const torch::Tensor& back_guides_t,
        const torch::Tensor& posts_t,
        size_t max_beam_width,
        float beam_cut,
        float fixed_stay_score,
        float q_shift,
        float q_scale,
        float byte_score_scale);
