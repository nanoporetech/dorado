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
    float pore_thr = 170.;
    size_t pore_cl_dist = 1000;
    size_t flank_size = 1000;
    //FIXME should probably be fractional
    //currently has to account for the adapter
    int flank_edist = 150;
    size_t adapter_length = 20;
    uint8_t adapter_edist = 3; //TODO figure out good threshold
    uint8_t pore_adapter_gap_thr = 30; //bp TODO figure out good threshold
    uint64_t expect_adapter_prefix = 200;
    //FIXME configure
    //TAIL_ADAPTER = 'GCAATACGTAACTGAACGAAGT'
    //HEAD_ADAPTER = 'AATGTACTTCGTTCAGTTACGTATTGCT'
    std::string adapter = "AATGTACTTCGTTCAGTTACGTATTGCT";
};

// struct ReadRange {
//     size_t start_pos;
//     size_t end_pos;
//     size_t start_sample;
//     size_t end_sample;

//     ReadRange(size_t start_pos,
//             size_t end_pos,
//             size_t start_sample,
//             size_t end_sample):
//             start_pos(start_pos), end_pos(end_pos),
//             start_sample(start_sample), end_sample(end_sample) {
//     }
// };

class DuplexSplitNode : public MessageSink {
public:
    typedef std::pair<uint64_t, uint64_t> PosRange;

    DuplexSplitNode(MessageSink& sink, DuplexSplitSettings settings,
                    int num_worker_threads = 5, size_t max_reads = 1000);
    ~DuplexSplitNode();

    std::vector<DuplexSplitNode::PosRange> possible_pore_regions(const Read& read);
    std::vector<PosRange> identify_splits(const Read& read);
    std::vector<Message> split(const Read& message, const std::vector<PosRange> &interspace_regions);

private:
    void worker_thread();  // Worker thread performs scaling and trimming asynchronously.
    MessageSink& m_sink;  // MessageSink to consume scaled reads.

    const DuplexSplitSettings m_settings;
    const int m_num_worker_threads;
    std::vector<std::unique_ptr<std::thread>> worker_threads;
};

}  // namespace dorado