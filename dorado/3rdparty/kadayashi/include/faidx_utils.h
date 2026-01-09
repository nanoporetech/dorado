#pragma once

#include "hts_types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace kadayashi::hts_utils {

std::string fetch_seq(const faidx_t* fai, const std::string& region);

int32_t fetch_seq_len(const faidx_t* fai, const std::string& seq_name);

std::vector<uint8_t> fetch_qual(const faidx_t* fai, const std::string& seq_name);

}  // namespace kadayashi::hts_utils
