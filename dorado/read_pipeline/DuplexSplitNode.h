#pragma once
#include "ReadPipeline.h"

namespace dorado {

struct DuplexSplitSettings {
    bool enabled = true;
    bool simplex_mode;
    float pore_thr = 160.;
    size_t pore_cl_dist = 4000;  // TODO maybe use frequency * 1sec here?
    float relaxed_pore_thr = 150.;
    //usually template read region to the left of potential spacer region
    size_t end_flank = 1200;
    //trim potentially erroneous (and/or PCR adapter) bases at end of query
    size_t end_trim = 200;
    //adjusted to adapter presense and potential loss of bases on query, leading to 'shift'
    size_t start_flank = 1700;
    int flank_edist = 150;
    int relaxed_flank_edist = 250;
    int adapter_edist = 4;
    int relaxed_adapter_edist = 6;
    uint64_t pore_adapter_range = 100;  //bp
    //in bases
    uint64_t expect_adapter_prefix = 200;
    //in samples
    uint64_t expect_pore_prefix = 5000;
    int middle_adapter_search_span = 1000;

    //TAIL_ADAPTER = 'GCAATACGTAACTGAACGAAGT'
    //HEAD_ADAPTER = 'AATGTACTTCGTTCAGTTACGTATTGCT'
    //clipped 4 letters from the beginning of head adapter (24 left)
    std::string adapter = "TACTTCGTTCAGTTACGTATTGCT";

    explicit DuplexSplitSettings(bool simplex_mode = false) : simplex_mode(simplex_mode) {}
};

class DuplexSplitNode : public MessageSink {
public:
    //TODO consider precumputing and reusing ranges with high signal
    struct ExtRead {
        std::shared_ptr<Read> read;
        torch::Tensor data_as_float32;
        std::vector<uint64_t> move_sums;

        explicit ExtRead(std::shared_ptr<Read> r);
    };

    typedef std::pair<uint64_t, uint64_t> PosRange;
    typedef std::vector<PosRange> PosRanges;

    DuplexSplitNode(MessageSink& sink,
                    DuplexSplitSettings settings,
                    int num_worker_threads = 5,
                    size_t max_reads = 1000);
    ~DuplexSplitNode();

private:
    typedef std::function<PosRanges(const ExtRead&)> SplitFinderF;

    std::vector<PosRange> possible_pore_regions(const ExtRead& read, float pore_thr) const;
    bool check_nearby_adapter(const Read& read, PosRange r, int adapter_edist) const;
    bool check_flank_match(const Read& read, PosRange r, int dist_thr) const;
    std::optional<PosRange> identify_extra_middle_split(const Read& read) const;

    std::vector<std::shared_ptr<Read>> split(std::shared_ptr<Read> read,
                                             const PosRanges& spacers) const;

    std::vector<std::pair<std::string, SplitFinderF>> build_split_finders() const;

    void worker_thread();  // Worker thread performs scaling and trimming asynchronously.
    MessageSink& m_sink;   // MessageSink to consume scaled reads.

    const DuplexSplitSettings m_settings;
    std::vector<std::pair<std::string, SplitFinderF>> m_split_finders;
    const int m_num_worker_threads;
    std::vector<std::unique_ptr<std::thread>> worker_threads;
};

}  // namespace dorado