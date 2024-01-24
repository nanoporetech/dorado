#pragma once

#include <ATen/core/TensorBody.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace dorado::basecall::decode {
std::tuple<std::string, std::string, std::vector<uint8_t>> beam_search_decode(
        const at::Tensor& scores_t,
        const at::Tensor& back_guides_t,
        const at::Tensor& posts_t,
        size_t max_beam_width,
        float beam_cut,
        float fixed_stay_score,
        float q_shift,
        float q_scale,
        float byte_score_scale);
}  // namespace dorado::basecall::decode
