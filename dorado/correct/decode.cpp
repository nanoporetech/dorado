#include "decode.h"

#include "conversions.h"
#include "types.h"

#include <spdlog/spdlog.h>

#include <array>

#ifdef NDEBUG
#define LOG_TRACE(...)
#else
#define LOG_TRACE(...) spdlog::trace(__VA_ARGS__)
#endif

struct base_count_t {
    int count = 0;
    int base;
};

struct PairHash {
    size_t operator()(const std::pair<int, int>& p) const {
        return std::hash<int>()(p.first) ^ std::hash<int>()(p.second);
    }
};
struct PairEqual {
    bool operator()(const std::pair<int, int>& p1, const std::pair<int, int>& p2) const {
        return p1.first == p2.first && p1.second == p2.second;
    }
};

namespace dorado::correction {

// The decode algorithm is split into 2 parts -
// 1. For positions that are predicted by the model (determined by
// the 'supported' indices), the model output is directly taken as
// the correct base.
// 2. For other positions, a majority vote is taken across the bases
// in that column. The majority base must have at least 2 reads
// supporting it. Now if the top 2 candidate bases have the same count and
// one of them matches the base in the target read, then the target read base is
// kept. Otherwise one of the two candidates is arbitrarily picked.
std::string decode_window(const WindowFeatures& wf) {
    std::string corrected_seq = "";

    auto encoding_to_idx_map = []() {
        std::array<int, 10> map = {0};
        map[0] = 0;
        map[1] = 1;
        map[2] = 2;
        map[3] = 3;
        map[4] = 4;
        map[5] = 0;
        map[6] = 1;
        map[7] = 2;
        map[8] = 3;
        map[9] = 4;
        return map;
    };
    static auto encoding_to_idx = encoding_to_idx_map();
    static auto base_decoding = gen_base_decoding();
    static auto base_forward = base_forward_mapping();

    if (wf.n_alns < 2) {
        return corrected_seq;
    }

    std::unordered_map<std::pair<int, int>, char, PairHash, PairEqual> bases_map;

    for (size_t i = 0; i < wf.supported.size(); i++) {
        LOG_TRACE("supported positions {},{} for {}", wf.supported[i].first, wf.supported[i].second,
                  wf.inferred_bases[i]);
        bases_map.insert({wf.supported[i], wf.inferred_bases[i]});
    }
    auto& bases = wf.bases;
    int tpos = -1, ins = 0;
    int length = (int)bases.sizes()[1];
    int* bases_tensor = bases.data_ptr<int>();
    for (int c = 0; c < length; c++) {
        const auto tbase = bases_tensor[c];
        if (base_decoding[tbase] == '*') {
            ins += 1;
        } else {
            tpos += 1;
            ins = 0;
        }

        auto p = std::make_pair(tpos, ins);
        auto found_p = bases_map.find(p);
        if (found_p != bases_map.end()) {
            auto new_base = found_p->second;
            if (new_base != '*') {
                corrected_seq += new_base;
                LOG_TRACE("{} tbase {} inferred base {} at {} {}", c, tbase, new_base, tpos, ins);
            }
        } else {
            std::array<base_count_t, 5> counter;
            for (int r = 0; r < wf.n_alns + 1; r++) {
                auto base = bases_tensor[r * length + c];
                if (base_decoding[base] == '.') {
                    continue;
                }
                auto idx = encoding_to_idx[base];
                counter[idx].base = base;
                counter[idx].count++;
            }

            std::sort(counter.begin(), counter.end(),
                      [](base_count_t a, base_count_t b) { return a.count > b.count; });

            auto& first = counter[0];
            auto& second = counter[1];

            char new_base;
            if ((first.count < 2) || (first.count == second.count &&
                                      (encoding_to_idx[first.base] == encoding_to_idx[tbase] ||
                                       encoding_to_idx[second.base] == encoding_to_idx[tbase]))) {
                new_base = base_decoding[tbase];
            } else {
                new_base = base_decoding[first.base];
            }

            new_base = base_forward[new_base];
            if (new_base != '*') {
                LOG_TRACE("{} tbase {} new base {}", c, tbase, new_base);
                corrected_seq += new_base;
            } else {
                LOG_TRACE("{} tbase {} new base {} skipping", c, tbase, new_base);
            }
        }
    }

    return corrected_seq;
}

}  // namespace dorado::correction
