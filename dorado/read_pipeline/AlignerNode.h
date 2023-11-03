#pragma once
#include "IndexFileAccess.h"
#include "Minimap2Options.h"
#include "ReadPipeline.h"
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
    // header_sequence_records is populated by the constructor.
    AlignerNode(const std::string& filename,
                const alignment::Minimap2Options& options,
                int threads);
    AlignerNode(std::shared_ptr<alignment::IndexFileAccess> index_file_access, int threads);
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
    size_t m_threads;
    std::vector<std::thread> m_workers;
    std::unique_ptr<alignment::AlignerImpl> m_aligner{};
    std::shared_ptr<alignment::IndexFileAccess> m_index_file_access{};
};

}  // namespace dorado
