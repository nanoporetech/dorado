#pragma once

#include "types.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace kadayashi {
bool variant_graph_gen(chunk_t &ck);
void variant_graph_propogate(chunk_t &ck);
int variant_graph_check_if_phasing_succeeded(const chunk_t &ck);
void variant_graph_haptag_reads(chunk_t &ck);
void variant_graph_do_simple_haptag(chunk_t &ck, const uint32_t n_iter_requested);

void normalize_readtaggings_ht(std::vector<std::unordered_map<uint32_t, uint8_t>> &arr_read2hp,
                               std::unordered_map<uint32_t, uint8_t> &breakpoint_reads,
                               const chunk_t &ck);

std::vector<uint8_t> variant_graph_do_simple_haptag1(chunk_t &ck, const uint32_t seedreadID);
std::unordered_map<uint32_t, uint8_t> variant_graph_do_simple_haptag1_give_ht(chunk_t &ck,
                                                                              const int seedreadID);
}  // namespace kadayashi
