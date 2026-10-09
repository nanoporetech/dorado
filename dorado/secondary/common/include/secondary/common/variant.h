#pragma once

#include <array>
#include <cstdint>
#include <iosfwd>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dorado::secondary {

struct FilterSpec {
    std::string_view id;
    std::string_view description;

    std::string header_string() const;
};

// "PASS" is a reserved filter string in the VCF specification, it should not be changed.
// Other filter strings are locally defined in the VCF header so can be changed if needed.
inline constexpr std::array KNOWN_FILTERS{
        FilterSpec{"PASS", "All filters passed"},
        FilterSpec{"LowQual", "Variant QUAL is below threshold"},
        FilterSpec{"IncompatZyg", "Predicted zygosity incompatible with ploidy"},
        FilterSpec{".", "Non-variant position"},
        FilterSpec{"LowGQ", "Variant GQ is below threshold"},
        FilterSpec{"LowAQ", "Variant AQ is below threshold"},
        FilterSpec{"LowVQ", "Variant VQ is below threshold"},
};

class Filters {
public:
    using FilterType = uint8_t;
    enum Filter : FilterType {
        Missing = 0,
        Pass = 1 << 0,
        LowQual = 1 << 1,
        IncompatZyg = 1 << 2,
        NonVariant = 1 << 3,
        LowGQ = 1 << 4,
        LowAQ = 1 << 5,
        LowVQ = 1 << 6,
    };

    Filters() = default;
    Filters(FilterType filter);
    void update(const Filter filter);
    std::vector<std::string_view> as_strings() const;

    bool operator==(const Filters&) const = default;
    bool operator&(const Filter& filter) const;
    bool operator|(const Filter& filter) const;

private:
    FilterType m_filter = Filter::Missing;
};

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
    Filters filters;
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

std::vector<float> compute_VAF_from_AD(const Variant& var, int32_t decimal_places);

}  // namespace dorado::secondary
