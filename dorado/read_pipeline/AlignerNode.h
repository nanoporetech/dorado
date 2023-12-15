#pragma once
#include "ReadPipeline.h"
#include "alignment/IndexFileAccess.h"
#include "alignment/Minimap2Options.h"
#include "utils/stats.h"
#include "utils/types.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

struct bam1_t;
typedef struct mm_tbuf_s mm_tbuf_t;

namespace dorado {

namespace alignment {
class Minimap2Index;
}  // namespace alignment

class AlignerNode : public MessageSink {
public:
    AlignerNode(std::shared_ptr<alignment::IndexFileAccess> index_file_access,
                const std::string& filename,
                const alignment::Minimap2Options& options,
                int threads);
    AlignerNode(std::shared_ptr<alignment::IndexFileAccess> index_file_access, int threads);
    ~AlignerNode();
    std::string get_name() const override { return "AlignerNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions&) override { terminate_impl(); }
    void restart() override;

    alignment::HeaderSequenceRecords get_sequence_records_for_header() const;

private:
    void start_threads();
    void terminate_impl();
    void worker_thread();
    std::shared_ptr<const alignment::Minimap2Index> get_index(const ReadCommon& read_common);
    void align_read_common(ReadCommon& read_common, mm_tbuf_t* tbuf);

    size_t m_threads;
    std::vector<std::thread> m_workers;
    std::shared_ptr<const alignment::Minimap2Index> m_index_for_bam_messages{};
    std::shared_ptr<alignment::IndexFileAccess> m_index_file_access{};
};

}  // namespace dorado
