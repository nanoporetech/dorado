#pragma once
#include "ReadPipeline.h"
#include "utils/stats.h"
#include "utils/types.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

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

class DuplexSplitNode : public MessageSink {
public:
    typedef std::pair<uint64_t, uint64_t> PosRange;
    typedef std::vector<PosRange> PosRanges;

    DuplexSplitNode(DuplexSplitSettings settings,
                    int num_worker_threads = 5,
                    size_t max_reads = 1000);
    ~DuplexSplitNode() { terminate_impl(); }
    std::string get_name() const override { return "DuplexSplitNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions& flush_options) override { terminate_impl(); }
    void restart() override;

    std::vector<std::shared_ptr<Read>> split(std::shared_ptr<Read> init_read) const;

private:
    void start_threads();
    void terminate_impl();
    //TODO consider precomputing and reusing ranges with high signal
    struct ExtRead {
        std::shared_ptr<Read> read;
        torch::Tensor data_as_float32;
        std::vector<uint64_t> move_sums;
        PosRanges possible_pore_regions;
    };

    typedef std::function<PosRanges(const ExtRead&)> SplitFinderF;

    ExtRead create_ext_read(std::shared_ptr<Read> r) const;
    std::vector<PosRange> possible_pore_regions(const ExtRead& read) const;
    bool check_nearby_adapter(const Read& read, PosRange r, int adapter_edist) const;
    std::optional<std::pair<PosRange, PosRange>> check_flank_match(const Read& read,
                                                                   PosRange r,
                                                                   float err_thr) const;
    std::optional<PosRange> identify_middle_adapter_split(const Read& read) const;
    std::optional<PosRange> identify_extra_middle_split(const Read& read) const;

    std::vector<std::shared_ptr<Read>> subreads(std::shared_ptr<Read> read,
                                                const PosRanges& spacers) const;

    std::vector<std::pair<std::string, SplitFinderF>> build_split_finders() const;

    void worker_thread();  // Worker thread performs splitting asynchronously.

    const DuplexSplitSettings m_settings;
    std::vector<std::pair<std::string, SplitFinderF>> m_split_finders;
    const int m_num_worker_threads;
    std::vector<std::unique_ptr<std::thread>> m_worker_threads;
};

}  // namespace dorado
