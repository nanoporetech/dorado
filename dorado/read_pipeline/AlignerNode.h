#pragma once

#include "ClientInfo.h"
#include "MessageSink.h"
#include "alignment/BedFileAccess.h"
#include "alignment/IndexFileAccess.h"
#include "alignment/Minimap2Options.h"
#include "utils/concurrency/async_task_executor.h"
#include "utils/concurrency/task_priority.h"
#include "utils/stats.h"
#include "utils/types.h"

#include <memory>
#include <string>
#include <vector>

struct bam1_t;
typedef struct mm_tbuf_s mm_tbuf_t;

namespace dorado {

namespace utils::concurrency {
class MultiQueueThreadPool;
}  // namespace utils::concurrency

namespace alignment {
class Minimap2Index;
}  // namespace alignment

class AlignerNode : public MessageSink {
public:
    AlignerNode(std::shared_ptr<alignment::IndexFileAccess> index_file_access,
                std::shared_ptr<alignment::BedFileAccess> bed_file_access,
                const std::string& index_file,
                const std::string& bed_file,
                const alignment::Minimap2Options& options,
                int threads);
    AlignerNode(std::shared_ptr<alignment::IndexFileAccess> index_file_access,
                std::shared_ptr<alignment::BedFileAccess> bed_file_access,
                std::shared_ptr<utils::concurrency::MultiQueueThreadPool> thread_pool,
                utils::concurrency::TaskPriority pipeline_priority);
    ~AlignerNode() { stop_input_processing(); }
    std::string get_name() const override { return "AlignerNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions&) override;
    void restart() override;

    alignment::HeaderSequenceRecords get_sequence_records_for_header() const;

private:
    void input_thread_fn();
    std::shared_ptr<const alignment::Minimap2Index> get_index(const ClientInfo& client_info);
    std::shared_ptr<dorado::alignment::BedFile> get_bedfile(const ClientInfo& client_info,
                                                            const std::string& bedfile);
    template <typename READ>
    void align_read(READ&& read);

    void align_bam_message(BamMessage&& bam_message);

    void align_read_common(ReadCommon& read_common, mm_tbuf_t* tbuf);
    void add_bed_hits_to_record(const std::string& genome, bam1_t* record);

    std::shared_ptr<utils::concurrency::MultiQueueThreadPool> m_thread_pool{};
    utils::concurrency::TaskPriority m_pipeline_priority{utils::concurrency::TaskPriority::normal};
    std::shared_ptr<const alignment::Minimap2Index> m_index_for_bam_messages{};
    std::shared_ptr<const alignment::BedFile> m_bedfile_for_bam_messages{};
    std::vector<std::string> m_header_sequence_names{};
    std::shared_ptr<alignment::IndexFileAccess> m_index_file_access{};
    std::shared_ptr<alignment::BedFileAccess> m_bed_file_access{};
    utils::concurrency::AsyncTaskExecutor m_task_executor;
};

}  // namespace dorado
