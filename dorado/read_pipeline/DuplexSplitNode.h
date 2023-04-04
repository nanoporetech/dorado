#pragma once
#include "ReadPipeline.h"

/*
To ask:
. does mod calling after splitting?
. separate reads pairing
. why do we keep trimmed?
. where is end_reason internally? -- put in a PR (modify dataloader)

To update:
. end_reason
. signal
. basecalls
. trims
. Chunks??? num_chunks_called
. num_trimmed samples???
. base mod info?
. Attributes
. shift/scale/scaling
. assert empty mappings
*/
namespace dorado {

/*
Available:
 . signal spikes
 . adapter sequences
 . self-folding

Q: is the signal scaled at this point?
Should I scale it back to detect 170pAmp spikes?

Plan:
I. Adapter-less mode
1. Detect spikes
2. For each detected spike try complement matching (exclude adapter) and split at spike
3. Try matching ends of the complete read (stepping away to account for adapter)
4. If match -- split somewhere in the middle

II. Adapter-aware mode
1. Detect spikes and likely adapter injections
2. If see both spike and adapter -- split (also handles non-duplex case)
3. If see one -- try matching the flanking regions
4. Try matching ends of the complete read (stepping away to account for adapter)
5. If match -- split somewhere in the middle
    Maybe try finding remnants of an adapter here?
*/
struct DuplexSplitSettings {
    float pore_thr = 160.;
    size_t pore_cl_dist = 4000; // TODO maybe use frequency * 1sec here?
    float relaxed_pore_thr = 150.;
    //usually template read region to the left of potential spacer region
    //FIXME rename to end_flank?!!
    size_t query_flank = 1200;
    //trim potentially erroneous (and/or PCR adapter) bases at end of query
    size_t query_trim = 200;
    //adjusted to adapter presense and potential loss of bases on query, leading to 'shift'
    //FIXME rename to start_flank?!!
    size_t target_flank = 1700;
    //FIXME should probably be fractional
    //currently has to account for the adapter
    int flank_edist = 150;
    //FIXME do we need both of them?
    int relaxed_flank_edist = 250;
    int adapter_edist = 4;
    int relaxed_adapter_edist = 6;
    uint64_t pore_adapter_range = 100; //bp TODO figure out good threshold
    //in bases
    uint64_t expect_adapter_prefix = 200;
    //in samples
    uint64_t expect_pore_prefix = 5000;
    //TAIL_ADAPTER = 'GCAATACGTAACTGAACGAAGT'
    //HEAD_ADAPTER = 'AATGTACTTCGTTCAGTTACGTATTGCT'
    //clipped 4 letters from the beginning of head adapter! 24 left
    std::string adapter = "TACTTCGTTCAGTTACGTATTGCT";
};

class DuplexSplitNode : public MessageSink {
public:
    typedef std::pair<uint64_t, uint64_t> PosRange;
    typedef std::vector<PosRange> PosRanges;
    typedef std::function<PosRanges (const Read&)> SplitFinderF;

    DuplexSplitNode(MessageSink& sink, DuplexSplitSettings settings,
                    int num_worker_threads = 5, size_t max_reads = 1000);
    ~DuplexSplitNode();

private:
    std::vector<PosRange> possible_pore_regions(const Read& read, float pore_thr) const;
    bool check_nearby_adapter(const Read& read, PosRange r, int adapter_edist) const;
    bool check_flank_match(const Read& read, PosRange r, int dist_thr) const;
    std::optional<PosRange> identify_extra_middle_split(const Read& read) const;

    std::vector<std::shared_ptr<Read>>
    split(std::shared_ptr<Read> read, const PosRanges& spacers) const;

    std::vector<std::pair<std::string, SplitFinderF>>
    build_split_finders() const;

    void worker_thread();  // Worker thread performs scaling and trimming asynchronously.
    MessageSink& m_sink;  // MessageSink to consume scaled reads.

    const DuplexSplitSettings m_settings;
    const int m_num_worker_threads;
    std::vector<std::unique_ptr<std::thread>> worker_threads;
};

}  // namespace dorado