#pragma once
#include "read_pipeline/ReadPipeline.h"
#include "utils/stats.h"
#include "utils/types.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

struct bam1_t;

namespace dorado {

namespace alignment {
class AlignerImpl;

// Exposed for testability
extern const std::string UNMAPPED_SAM_LINE_STRIPPED;

}  // namespace alignment

class AlignerNode : public MessageSink {
public:
    struct Minimap2Options {
        short kmer_size;
        short window_size;
        uint64_t index_batch_size;
        bool print_secondary;
        int best_n_secondary;
        int bandwidth;
        int bandwidth_long;
        bool soft_clipping;
        bool secondary_seq;
        bool print_aln_seq;
    };
    static constexpr Minimap2Options dflt_options{15,  10,    16000000000ull, true,  5,
                                                  500, 20000, false,          false, false};

public:
    // header_sequence_records is populated by the constructor.
    AlignerNode(const std::string& filename, const Minimap2Options& options, int threads);
    ~AlignerNode();
    std::string get_name() const override { return "AlignerNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions& flush_options) override { terminate_impl(); }
    void restart() override;

    using bam_header_sq_t = std::vector<std::pair<char*, uint32_t>>;
    bam_header_sq_t get_sequence_records_for_header() const;

private:
    void start_threads();
    void terminate_impl();
    void worker_thread();
    size_t m_threads{1};
    std::vector<std::thread> m_workers;
    std::unique_ptr<alignment::AlignerImpl> m_aligner;
};

}  // namespace dorado
