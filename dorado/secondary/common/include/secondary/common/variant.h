#pragma once

#include <cstdint>
#include <iosfwd>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dorado::secondary {

struct Variant {
    int32_t seq_id = -1;
    int64_t pos = -1;
    std::string ref;
    std::vector<std::string> alts;
    std::string filter;
    std::unordered_map<std::string, std::string> info;
    float qual = 0.0f;
    std::unordered_map<std::string, std::string> format;
    int64_t rstart = 0;
    int64_t rend = 0;
};

std::ostream& operator<<(std::ostream& os, const Variant& v);

bool operator==(const Variant& lhs, const Variant& rhs);

bool operator<(const Variant& lhs, const Variant& rhs);

bool is_valid(const Variant& var);

bool is_reference_record(const Variant& var);

int64_t variant_end(const Variant& var);

bool variant_ends_before_position(const Variant& var, int32_t seq_id, int64_t pos);

bool variant_covers_position(const Variant& var, int32_t seq_id, int64_t pos);

void set_gvcf_reference_block_end(Variant& var, int64_t end);

float get_gvcf_reference_record_gq_margin(float gq,
                                          std::span<const std::pair<int64_t, float>> margins);

}  // namespace dorado::secondary
