#pragma once

#include <cstdint>
#include <tuple>

namespace dorado::alignment {

struct Minimap2IndexOptions {
    short kmer_size;
    short window_size;
    uint64_t index_batch_size;
    std::string mm2_preset;
};

inline bool operator<(const Minimap2IndexOptions& l, const Minimap2IndexOptions& r) {
    return std::tie(l.kmer_size, l.window_size, l.index_batch_size) <
           std::tie(r.kmer_size, r.window_size, r.index_batch_size);
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
    return std::tie(l.kmer_size, l.window_size, l.index_batch_size) ==
           std::tie(r.kmer_size, r.window_size, r.index_batch_size);
}

inline bool operator!=(const Minimap2IndexOptions& l, const Minimap2IndexOptions& r) {
    return !(l == r);
}

struct Minimap2MappingOptions {
    int best_n_secondary;
    int bandwidth;
    int bandwidth_long;
    bool soft_clipping;
    bool secondary_seq;
    bool print_secondary;
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
    bool print_aln_seq;
};

inline bool operator==(const Minimap2Options& l, const Minimap2Options& r) {
    return static_cast<Minimap2IndexOptions>(l) == static_cast<Minimap2IndexOptions>(r) &&
           static_cast<Minimap2MappingOptions>(l) == static_cast<Minimap2MappingOptions>(r);
}

inline bool operator!=(const Minimap2Options& l, const Minimap2Options& r) { return !(l == r); }

static const Minimap2Options dflt_options{{15, 10, 16000000000ull, "lr:hq"},
                                          {5, 500, 20000, false, false, true},
                                          false};
}  // namespace dorado::alignment
