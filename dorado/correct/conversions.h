#pragma once

#include <array>

namespace dorado::correction {

const float MIN_QSCORE = 33.f;
const float MAX_QSCORE = 126.f;

float normalize_quals(float q);

std::array<char, 128> base_forward_mapping();

std::array<int, 128> gen_base_encoding();

std::array<char, 11> gen_base_decoding();

}  // namespace dorado::correction
