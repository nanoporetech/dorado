#pragma once

#include "conversions.h"
#include "types.h"

#include <array>

namespace dorado::correction {

std::vector<std::string> decode_windows(const std::vector<WindowFeatures>& wfs) {
    std::vector<std::string> corrected_reads;
    std::string corrected_seq;

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

    std::unordered_map<std::pair<int, int>, char, PairHash, PairEqual> bases_map;

    for (const auto& wf : wfs) {
        if (wf.n_alns < 2) {
            if (corrected_seq.length() > 0) {
                corrected_reads.push_back(corrected_seq);
                //spdlog::info("added seq naln < 2 of len {}", corrected_seq.length());
                corrected_seq = "";
            }

            continue;
        }

        bases_map.clear();
        for (size_t i = 0; i < wf.supported.size(); i++) {
            //spdlog::info("supported positions {},{} for {}", wf.supported[i].first, wf.supported[i].second, wf.inferred_bases[i]);
            bases_map.insert({wf.supported[i], wf.inferred_bases[i]});
        }
        auto& bases = wf.bases;
        int tpos = -1, ins = 0;
        int length = bases.sizes()[1];
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
                    //spdlog::info("{} tbase {} inferred base {} at {} {}", c, tbase, new_base, tpos, ins);
                }
            } else {
                std::array<base_count_t, 5> counter;
                for (int r = 0; r < wf.n_alns + 1; r++) {
                    auto base = bases_tensor[r * length + c];
                    if (base_decoding[base] == '.') {
                        continue;
                    }
                    auto idx = encoding_to_idx[base];
                    counter[idx].b = base;
                    counter[idx].c++;
                }

                std::sort(counter.begin(), counter.end(),
                          [](base_count_t a, base_count_t b) { return a.c > b.c; });

                auto& first = counter[0];
                auto& second = counter[1];

                char new_base;
                if ((first.c < 2) || (first.c == second.c &&
                                      (encoding_to_idx[first.b] == encoding_to_idx[tbase] ||
                                       encoding_to_idx[second.b] == encoding_to_idx[tbase]))) {
                    new_base = base_decoding[tbase];
                } else {
                    new_base = base_decoding[first.b];
                }

                new_base = base_forward[new_base];
                if (new_base != '*') {
                    //spdlog::info("{} tbase {} new base {}", c, tbase, new_base);
                    corrected_seq += new_base;
                } else {
                    //spdlog::info("{} tbase {} new base {} skipping", c, tbase, new_base);
                }
            }
        }
    }

    if (!corrected_seq.empty()) {
        //spdlog::info("added seq end of len {}", corrected_seq.length());
        corrected_reads.push_back(corrected_seq);
    }

    return corrected_reads;
}
}  // namespace dorado::correction
