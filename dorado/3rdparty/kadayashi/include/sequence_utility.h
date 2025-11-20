#pragma once

#include "types.h"

#include <cstdint>
#include <vector>

namespace kadayashi {

void seq2nt4seq(const char *seq, const int seq_l, std::vector<uint8_t> &h);
std::string nt4seq2seq(const std::vector<uint8_t> &allele_vec);
int diff_of_top_two(std::vector<uint32_t> &d);
uint32_t max_of_u32_vec(const std::vector<uint32_t> &d, int *idx);

}  // namespace kadayashi
