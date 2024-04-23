#pragma once

#include "ReadPipeline.h"
#include "alignment/Minimap2Aligner.h"
#include "alignment/Minimap2Index.h"
#include "alignment/Minimap2IndexSupportTypes.h"
#include "read_pipeline/MessageSink.h"
#include "read_pipeline/messages.h"
#include "utils/AsyncQueue.h"
#include "utils/stats.h"
#include "utils/types.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace dorado {

class ErrorCorrectionMapperNode : public MessageSink {
public:
    ErrorCorrectionMapperNode(const std::string& index_file, int threads);
    ~ErrorCorrectionMapperNode() = default;
    std::string get_name() const override { return "ErrorCorrectionMapperNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions&) override{};
    void restart() override {}
    // Main driver function.
    void process(Pipeline& pipeline);

private:
    std::string m_index_file;
    int m_num_threads;

    std::unique_ptr<alignment::Minimap2Aligner> m_aligner;
    std::shared_ptr<alignment::Minimap2Index> m_index;

    void input_thread_fn();
    void load_read_fn();

    void extract_alignments(const mm_reg1_t* reg,
                            int hits,
                            const std::string& qread,
                            const std::string& qname,
                            const std::vector<uint8_t>& qqual);

    // Queue for reads being aligned.
    utils::AsyncQueue<BamPtr> m_reads_queue;

    // Map to collects alignments by target id.
    std::mutex m_correction_mtx;
    std::unordered_map<std::string, CorrectionAlignments> m_correction_records;

    // Mutex per target id to prevent a global lock across all targets.
    std::unordered_map<std::string, std::unique_ptr<std::mutex>> m_read_mutex;

    std::unique_ptr<std::thread> m_reader_thread;
    std::vector<std::unique_ptr<std::thread>> m_aligner_threads;

    std::atomic<int> m_reads_read{0};
    std::atomic<int> m_alignments_processed{0};
};

}  // namespace dorado
