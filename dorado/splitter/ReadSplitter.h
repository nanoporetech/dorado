#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace dorado {

class SimplexRead;
using SimplexReadPtr = std::unique_ptr<SimplexRead>;

namespace splitter {

struct RNASplitSettings {
    int16_t pore_thr = 1500;
    uint64_t pore_cl_dist = 2000;
    //maximal 'open pore' region to consider (bp)
    uint64_t max_pore_region = 500;
    //in samples
    uint64_t expect_pore_prefix = 2000;
};

struct DuplexSplitSettings {
    bool enabled = true;
    bool simplex_mode = false;
    float pore_thr = 2.4f;
    uint64_t pore_cl_dist = 500;  // in samples
    //maximal 'open pore' region to consider (bp)
    uint64_t max_pore_region = 500;
    //only use position with signal maximal as a tentative open pore
    bool use_argmax = true;
    //number of bases to check quality (starting with pore region start)
    int qscore_check_span = 5;
    //only take fixed number of candidates with maximal signal
    int top_candidates = 10;
    //filter tentative open pore regions with mean qscore higher than threshold
    float mean_qscore_thr = 10.;
    //usually template read region to the left of potential spacer region
    uint64_t strand_end_flank = 1200;
    //trim potentially erroneous (and/or PCR adapter) bases at end of query
    uint64_t strand_end_trim = 200;
    //adjusted to adapter presense and potential loss of bases on query, leading to 'shift'
    uint64_t strand_start_flank = 1700;
    //minimal query size to consider in "short read" case
    uint64_t min_flank = 300;
    float flank_err = 0.15f;
    float relaxed_flank_err = 0.275f;
    int adapter_edist = 4;
    int relaxed_adapter_edist = 8;
    //bp from end of tentative pore to end of adapter
    //(~ max pore-adapter dist + adapter length)
    uint64_t pore_adapter_span = 50;
    //in bases
    uint64_t expect_adapter_prefix = 200;
    //in samples
    uint64_t expect_pore_prefix = 5000;
    uint64_t middle_adapter_search_span = 1000;
    float middle_adapter_search_frac = 0.2f;

    //TODO put in config
    //TODO use string_view into a shared adapter string when we have it
    //Adapter sequence we expect to see at the beginning of the read
    //Sequence below corresponds to the current 'head' adapter 'AATGTACTTCGTTCAGTTACGTATTGCT'
    // with 4bp clipped from the beginning (24bp left)
    std::string adapter = "TACTTCGTTCAGTTACGTATTGCT";

    explicit DuplexSplitSettings(bool pA_scaling) {
        if (pA_scaling) {
            pore_thr = 2.8f;
        }
    }
};

class ReadSplitter {
public:
    ReadSplitter() = default;
    virtual ~ReadSplitter() = default;

    virtual std::vector<SimplexReadPtr> split(SimplexReadPtr init_read) const = 0;
};

}  // namespace splitter

}  // namespace dorado
