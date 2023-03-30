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
    size_t pore_cl_dist = 1000;
    size_t templ_flank = 1100;
    //trim potentially erroneous (and/or PCR adapter) bases at end of template
    size_t templ_trim = 100;
    //adjusted to adapter presense and potential loss of bases on template, leading to 'shift'
    size_t compl_flank = 1200;
    //FIXME should probably be fractional
    //currently has to account for the adapter
    int flank_edist = 150;
    uint8_t adapter_edist = 3; //TODO figure out good threshold. 3 seems ok for hac/sup, but can probably relax
    uint8_t pore_adapter_gap_thr = 30; //bp TODO figure out good threshold
    //TODO don't need if we have more efficient adapter detection
    uint64_t expect_adapter_prefix = 200;
    //TAIL_ADAPTER = 'GCAATACGTAACTGAACGAAGT'
    //HEAD_ADAPTER = 'AATGTACTTCGTTCAGTTACGTATTGCT'
    //clipped 4 letters from the beginning of head adapter! 24 left
    std::string adapter = "TACTTCGTTCAGTTACGTATTGCT";
};

class DuplexSplitNode : public MessageSink {
public:
    typedef std::pair<uint64_t, uint64_t> PosRange;

    DuplexSplitNode(MessageSink& sink, DuplexSplitSettings settings,
                    int num_worker_threads = 5, size_t max_reads = 1000);
    ~DuplexSplitNode();

private:
    std::vector<PosRange> identify_splits(const Read& read);
    std::vector<PosRange> identify_splits_check_all(const Read& read);
    std::optional<PosRange> identify_extra_middle_split(const Read& read);
    std::vector<std::shared_ptr<Read>> split(const Read& message, const std::vector<PosRange> &interspace_regions);

    void worker_thread();  // Worker thread performs scaling and trimming asynchronously.
    MessageSink& m_sink;  // MessageSink to consume scaled reads.

    const DuplexSplitSettings m_settings;
    const int m_num_worker_threads;
    std::vector<std::unique_ptr<std::thread>> worker_threads;
};

}  // namespace dorado