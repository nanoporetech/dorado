#pragma once

#include "alignment/BedFileAccess.h"
#include "alignment/IndexFileAccess.h"
#include "alignment/Minimap2Options.h"
#include "hts_utils/header_sq_record.h"
#include "read_pipeline/base/ClientInfo.h"
#include "read_pipeline/base/MessageSink.h"
#include "utils/concurrency/AsyncExecutor.h"

#include <memory>
#include <string>
#include <vector>

struct bam1_t;
typedef struct mm_tbuf_s mm_tbuf_t;

namespace dorado {

namespace alignment {
class Minimap2Index;
}  // namespace alignment

class AlignerNode : public MessageSink {
public:
    static inline constexpr std::size_t MAX_INPUT_QUEUE_SIZE{10000};
    static inline constexpr std::size_t MAX_PROCESSING_QUEUE_SIZE{MAX_INPUT_QUEUE_SIZE / 2};
    static inline constexpr char QUEUE_THREAD_NAME[] = "aligner";

    AlignerNode(std::shared_ptr<alignment::IndexFileAccess> index_file_access,
                std::shared_ptr<alignment::BedFileAccess> bed_file_access,
                const std::string& index_file,
                const std::string& bed_file,
                const alignment::Minimap2Options& options,
                utils::concurrency::AsyncExecutor&& task_executor);
    AlignerNode(std::shared_ptr<alignment::IndexFileAccess> index_file_access,
                std::shared_ptr<alignment::BedFileAccess> bed_file_access,
                utils::concurrency::AsyncExecutor&& task_executor);
    ~AlignerNode();

    std::string get_name() const override;
    stats::NamedStats sample_stats() const override;
    void terminate(const TerminateOptions&) override;
    void restart() override;

    utils::HeaderSQRecords get_sequence_records_for_header() const;

private:
    AlignerNode(const AlignerNode&) = delete;
    AlignerNode& operator=(const AlignerNode&) = delete;
    AlignerNode(AlignerNode&&) = delete;
    AlignerNode& operator=(AlignerNode&&) = delete;

private:
    void input_thread_fn();
    std::shared_ptr<const alignment::Minimap2Index> get_index(const ClientInfo& client_info);
    std::shared_ptr<dorado::alignment::BedFile> get_bedfile(const ClientInfo& client_info,
                                                            const std::string& bedfile);
    template <typename Read>
    void align_read(Read&& read);

    void align_bam_message(BamMessage&& bam_message);

    void align_read_common(ReadCommon& read_common, mm_tbuf_t* tbuf);
    void add_bed_hits_to_record(const std::string& genome, bam1_t* record);

    std::shared_ptr<const alignment::Minimap2Index> m_index_for_bam_messages{};
    std::shared_ptr<const alignment::BedFile> m_bedfile_for_bam_messages{};
    std::vector<std::string> m_header_sequence_names{};
    std::shared_ptr<alignment::IndexFileAccess> m_index_file_access{};
    std::shared_ptr<alignment::BedFileAccess> m_bed_file_access{};
    utils::concurrency::AsyncExecutor m_task_executor;
};

}  // namespace dorado
