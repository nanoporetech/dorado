#pragma once

#include "demux/BarcodeClassifierSelector.h"
#include "read_pipeline/base/MessageSink.h"
#include "utils/concurrency/AsyncExecutor.h"

#include <atomic>
#include <map>
#include <mutex>
#include <string>

namespace dorado {

namespace demux {
struct BarcodingInfo;
}

class BarcodeClassifierNode final : public MessageSink {
public:
    static inline constexpr std::size_t MAX_INPUT_QUEUE_SIZE{10000};
    static inline constexpr std::size_t MAX_PROCESSING_QUEUE_SIZE{MAX_INPUT_QUEUE_SIZE / 2};
    static inline constexpr char QUEUE_THREAD_NAME[] = "barcoding";

    explicit BarcodeClassifierNode(utils::concurrency::AsyncExecutor&& task_executor);

    ~BarcodeClassifierNode();

    std::string get_name() const override;
    stats::NamedStats sample_stats() const override;
    void terminate(const TerminateOptions&) override;
    void restart() override;

private:
    BarcodeClassifierNode(const BarcodeClassifierNode&) = delete;
    BarcodeClassifierNode& operator=(const BarcodeClassifierNode&) = delete;
    BarcodeClassifierNode(BarcodeClassifierNode&&) = delete;
    BarcodeClassifierNode& operator=(BarcodeClassifierNode&&) = delete;

private:
    utils::concurrency::AsyncExecutor m_task_executor;

    std::atomic<int> m_num_records{0};
    demux::BarcodeClassifierSelector m_barcoder_selector{};

    void input_thread_fn();
    void barcode(BamMessage& read, const demux::BarcodingInfo* barcoding_info);
    void barcode(SimplexRead& read);

    // Track how many reads were classified as each barcode for debugging
    // purposes.
    std::atomic<size_t> m_mid_strand_count{0};
    std::map<std::string, size_t> m_barcode_count;
    mutable std::mutex m_barcode_count_mutex;
};

}  // namespace dorado
