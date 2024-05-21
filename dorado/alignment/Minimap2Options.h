#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>

namespace dorado::alignment {

constexpr inline std::string_view DEFAULT_MM_PRESET{"lr:hq"};

namespace minimap2 {
class IdxOptHolder;
class MapOptHolder;
}  // namespace minimap2

struct Minimap2IndexOptions {
    Minimap2IndexOptions();
    std::optional<short> kmer_size;
    std::optional<short> window_size;
    std::optional<uint64_t> index_batch_size;
    std::string mm2_preset{DEFAULT_MM_PRESET};  // By default we use a preset, hence not an optional
    std::shared_ptr<minimap2::IdxOptHolder> index_options;
    std::string junc_bed;
};

bool operator<(const Minimap2IndexOptions& l, const Minimap2IndexOptions& r);
bool operator>(const Minimap2IndexOptions& l, const Minimap2IndexOptions& r);
bool operator<=(const Minimap2IndexOptions& l, const Minimap2IndexOptions& r);
bool operator>=(const Minimap2IndexOptions& l, const Minimap2IndexOptions& r);
bool operator==(const Minimap2IndexOptions& l, const Minimap2IndexOptions& r);
bool operator!=(const Minimap2IndexOptions& l, const Minimap2IndexOptions& r);

struct Minimap2MappingOptions {
    Minimap2MappingOptions();
    std::optional<int> best_n_secondary;
    std::optional<int> bandwidth;
    std::optional<int> bandwidth_long;
    std::optional<bool> soft_clipping;
    bool secondary_seq = false;  // Not available to be set by the user, hence not optional
    std::optional<bool> print_secondary;
    std::optional<int> occ_dist;
    std::optional<int> min_chain_score;
    std::optional<int> zdrop;
    std::optional<int> zdrop_inv;
    std::optional<std::string> cs;
    std::optional<std::string> dual;
    std::optional<uint64_t> mini_batch_size;
    std::shared_ptr<minimap2::MapOptHolder> mapping_options;
};

bool operator<(const Minimap2MappingOptions& l, const Minimap2MappingOptions& r);
bool operator>(const Minimap2MappingOptions& l, const Minimap2MappingOptions& r);
bool operator<=(const Minimap2MappingOptions& l, const Minimap2MappingOptions& r);
bool operator>=(const Minimap2MappingOptions& l, const Minimap2MappingOptions& r);
bool operator==(const Minimap2MappingOptions& l, const Minimap2MappingOptions& r);
bool operator!=(const Minimap2MappingOptions& l, const Minimap2MappingOptions& r);

struct Minimap2Options : public Minimap2IndexOptions, public Minimap2MappingOptions {
    bool print_aln_seq;  // Not available to be set by the user, hence not optional
};

bool operator==(const Minimap2Options& l, const Minimap2Options& r);
bool operator!=(const Minimap2Options& l, const Minimap2Options& r);

Minimap2Options create_dflt_options();  // the default preset is "lr:hq"
Minimap2Options create_preset_options(const std::string& preset);

}  // namespace dorado::alignment
