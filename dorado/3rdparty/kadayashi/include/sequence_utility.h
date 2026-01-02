#pragma once

#include "types.h"

#include <cstdint>
#include <span>
#include <vector>

namespace kadayashi {

std::vector<uint8_t> seq2nt4seq(const std::string_view &seq);
std::string nt4seq2seq(const std::vector<uint8_t> &allele_vec);
bool max_of_u32_arr(const std::span<const uint32_t> &d, int *idx, uint32_t *val);

}  // namespace kadayashi
