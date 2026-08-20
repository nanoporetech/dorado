#pragma once

#include <cstdint>
#include <iosfwd>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dorado::secondary {

// "PASS" is a reserved filter string in the VCF specification, it should not be changed.
// Other filter strings are locally defined in the VCF header so can be changed if needed.
constexpr std::string_view VARIANT_PASS_FILTER{"PASS"};
constexpr std::string_view VARIANT_LOWQUAL_FILTER{"LowQual"};
constexpr std::string_view VARIANT_ZYGOSITY_FILTER{"IncompatZyg"};
constexpr std::string_view NO_VARIANT_FILTER{"."};

// Plain `const char*` (not `string_view`) so these convert implicitly to `std::string`
// map keys/values; `std::string`'s converting constructor from `string_view` is explicit.
constexpr const char* FORMAT_TAG_VARIANT_SOURCE{"SRC"};
constexpr const char* VARIANT_SOURCE_INFERENCE{"I"};
constexpr const char* VARIANT_SOURCE_KADAYASHI{"K"};
constexpr const char* VARIANT_SOURCE_REFERENCE{"R"};
constexpr const char* VARIANT_SOURCE_UNKNOWN{"."};

struct Variant {
    int32_t seq_id = -1;
    int64_t pos = -1;
    std::string ref;
    std::vector<std::string> alts;
    std::string_view filter;
    std::unordered_map<std::string, std::string> info;
    float qual = 0.0f;
    std::unordered_map<std::string, std::string> format;
    int64_t rstart = 0;
    int64_t rend = 0;
};

std::ostream& operator<<(std::ostream& os, const Variant& v);

bool operator==(const Variant& lhs, const Variant& rhs);

bool operator<(const Variant& lhs, const Variant& rhs);

std::string to_string(const Variant& v);

bool is_valid(const Variant& var);

bool is_reference_record(const Variant& var);

int64_t variant_end(const Variant& var);

bool variant_ends_before_position(const Variant& var, int32_t seq_id, int64_t pos);

bool variant_covers_position(const Variant& var, int32_t seq_id, int64_t pos);

void set_gvcf_reference_block_end(Variant& var, int64_t end);

float get_gvcf_reference_record_gq_margin(float gq,
                                          std::span<const std::pair<int64_t, float>> margins);

}  // namespace dorado::secondary
