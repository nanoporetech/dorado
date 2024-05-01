#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>

namespace dorado::alignment {

static const std::string DEFAULT_MM_PRESET{"lr:hq"};

struct Minimap2IndexOptions {
    std::optional<short> kmer_size;
    std::optional<short> window_size;
    std::optional<uint64_t> index_batch_size;
    std::string mm2_preset =
            DEFAULT_MM_PRESET;  // By default we use a preset, hence not an optional
};

inline bool operator<(const Minimap2IndexOptions& l, const Minimap2IndexOptions& r) {
    return std::tie(l.kmer_size, l.window_size, l.index_batch_size, l.mm2_preset) <
           std::tie(r.kmer_size, r.window_size, r.index_batch_size, r.mm2_preset);
}

inline bool operator>(const Minimap2IndexOptions& l, const Minimap2IndexOptions& r) {
    return r < l;
}

inline bool operator<=(const Minimap2IndexOptions& l, const Minimap2IndexOptions& r) {
    return !(l > r);
}

inline bool operator>=(const Minimap2IndexOptions& l, const Minimap2IndexOptions& r) {
    return !(l < r);
}

inline bool operator==(const Minimap2IndexOptions& l, const Minimap2IndexOptions& r) {
    return std::tie(l.kmer_size, l.window_size, l.index_batch_size, l.mm2_preset) ==
           std::tie(r.kmer_size, r.window_size, r.index_batch_size, r.mm2_preset);
}

inline bool operator!=(const Minimap2IndexOptions& l, const Minimap2IndexOptions& r) {
    return !(l == r);
}

struct Minimap2MappingOptions {
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
};

inline bool operator<(const Minimap2MappingOptions& l, const Minimap2MappingOptions& r) {
    return std::tie(l.best_n_secondary, l.bandwidth, l.bandwidth_long, l.soft_clipping,
                    l.secondary_seq, l.print_secondary) <
           std::tie(r.best_n_secondary, r.bandwidth, r.bandwidth_long, r.soft_clipping,
                    r.secondary_seq, r.print_secondary);
}

inline bool operator>(const Minimap2MappingOptions& l, const Minimap2MappingOptions& r) {
    return r < l;
}

inline bool operator<=(const Minimap2MappingOptions& l, const Minimap2MappingOptions& r) {
    return !(l > r);
}

inline bool operator>=(const Minimap2MappingOptions& l, const Minimap2MappingOptions& r) {
    return !(l < r);
}

inline bool operator==(const Minimap2MappingOptions& l, const Minimap2MappingOptions& r) {
    return std::tie(l.best_n_secondary, l.bandwidth, l.bandwidth_long, l.soft_clipping,
                    l.secondary_seq, l.print_secondary) ==
           std::tie(r.best_n_secondary, r.bandwidth, r.bandwidth_long, r.soft_clipping,
                    r.secondary_seq, r.print_secondary);
}

inline bool operator!=(const Minimap2MappingOptions& l, const Minimap2MappingOptions& r) {
    return !(l == r);
}

struct Minimap2Options : public Minimap2IndexOptions, public Minimap2MappingOptions {
    bool print_aln_seq;  // Not available to be set by the user, hence not optional
};

inline bool operator==(const Minimap2Options& l, const Minimap2Options& r) {
    return static_cast<const Minimap2IndexOptions&>(l) == r &&
           static_cast<const Minimap2MappingOptions&>(l) == r;
}

inline bool operator!=(const Minimap2Options& l, const Minimap2Options& r) { return !(l == r); }

static const Minimap2Options dflt_options{Minimap2IndexOptions{}, Minimap2MappingOptions{}, false};
}  // namespace dorado::alignment
