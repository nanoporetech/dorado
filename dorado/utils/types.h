#pragma once

#include <memory>
#include <string>

struct bam1_t;
struct mm_tbuf_s;

namespace dorado {

struct DuplexSplitSettings {
    bool enabled = true;
    bool simplex_mode = false;
    float pore_thr = 2.2;
    uint64_t pore_cl_dist = 4000;  // TODO maybe use frequency * 1sec here?
    //maximal 'open pore' region to consider (bp)
    uint64_t max_pore_region = 500;
    //usually template read region to the left of potential spacer region
    uint64_t strand_end_flank = 1200;
    //trim potentially erroneous (and/or PCR adapter) bases at end of query
    uint64_t strand_end_trim = 200;
    //adjusted to adapter presense and potential loss of bases on query, leading to 'shift'
    uint64_t strand_start_flank = 1700;
    //minimal query size to consider in "short read" case
    uint64_t min_flank = 300;
    float flank_err = 0.15;
    float relaxed_flank_err = 0.275;
    int adapter_edist = 4;
    int relaxed_adapter_edist = 8;
    uint64_t pore_adapter_range = 100;  //bp
    //in bases
    uint64_t expect_adapter_prefix = 200;
    //in samples
    uint64_t expect_pore_prefix = 5000;
    uint64_t middle_adapter_search_span = 1000;
    float middle_adapter_search_frac = 0.2;

    //TODO put in config
    //TODO use string_view into a shared adapter string when we have it
    //Adapter sequence we expect to see at the beginning of the read
    //Sequence below corresponds to the current 'head' adapter 'AATGTACTTCGTTCAGTTACGTATTGCT'
    // with 4bp clipped from the beginning (24bp left)
    std::string adapter = "TACTTCGTTCAGTTACGTATTGCT";
};

struct ReadGroup {
    std::string run_id;
    std::string basecalling_model;
    std::string flowcell_id;
    std::string device_id;
    std::string exp_start_time;
    std::string sample_id;
};

struct BamDestructor {
    void operator()(bam1_t *);
};
using BamPtr = std::unique_ptr<bam1_t, BamDestructor>;

struct MmTbufDestructor {
    void operator()(mm_tbuf_s *);
};
using MmTbufPtr = std::unique_ptr<mm_tbuf_s, MmTbufDestructor>;

enum class ReadOrder { UNRESTRICTED, BY_CHANNEL, BY_TIME };

inline std::string to_string(ReadOrder read_order) {
    switch (read_order) {
    case ReadOrder::UNRESTRICTED:
        return "UNRESTRICTED";
    case ReadOrder::BY_CHANNEL:
        return "BY_CHANNEL";
    case ReadOrder::BY_TIME:
        return "BY_TIME";
    default:
        return "Unknown";
    }
}

struct BaseModInfo {
    BaseModInfo(std::string alphabet_, std::string long_names_, std::string context_)
            : alphabet(std::move(alphabet_)),
              long_names(std::move(long_names_)),
              context(std::move(context_)) {}
    std::string alphabet;
    std::string long_names;
    std::string context;
};

}  // namespace dorado
