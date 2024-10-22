#pragma once
#include "MessageSink.h"
#include "utils/hts_file.h"
#include "utils/stats.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>

struct bam1_t;

namespace dorado {

class HtsWriter : public MessageSink {
public:
    HtsWriter(utils::HtsFile& file, std::string gpu_names);
    ~HtsWriter();
    std::string get_name() const override { return "HtsWriter"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions&) override;
    void restart() override {
        start_input_processing([this] { input_thread_fn(); }, "hts_writer");
    }

    int write(bam1_t* record);
    size_t get_total() const { return m_total; }
    size_t get_primary() const { return m_primary; }
    size_t get_unmapped() const { return m_unmapped; }

    static utils::HtsFile::OutputMode get_output_mode(const std::string& mode);

private:
    size_t m_total{0};
    size_t m_primary{0};
    size_t m_unmapped{0};
    size_t m_secondary{0};
    size_t m_supplementary{0};

    utils::HtsFile& m_file;

    std::string m_gpu_names{};

    void input_thread_fn();
    std::atomic<int> m_duplex_reads_written{0};
    std::atomic<int> m_split_reads_written{0};

    // Expected usage:
    //  single writer thread calling add()
    //  many threads may concurrently call size().
    class ProcessedReadIds {
        std::unordered_set<std::string> read_ids;
        std::atomic<std::size_t> m_threadsafe_count_of_reads{};

    public:
        // Thread safe access to count of unique read-ids
        std::size_t size() const;

        // Not thread safe for concurrent calls.
        void add(std::string read_id);
    } m_processed_read_ids;
};

}  // namespace dorado
