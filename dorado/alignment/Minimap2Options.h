#pragma once

#include <cstdint>
#include <tuple>

namespace dorado::alignment {

struct Minimap2IndexOptions {
    short kmer_size;
    short window_size;
    uint64_t index_batch_size;
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

struct Minimap2Options : public Minimap2IndexOptions, public Minimap2MappingOptions {
    bool print_aln_seq;
};
static constexpr Minimap2Options dflt_options{{15, 10, 16000000000ull},
                                              {true, 5, 500, 20000, false, false},
                                              false};
}  // namespace dorado::alignment
