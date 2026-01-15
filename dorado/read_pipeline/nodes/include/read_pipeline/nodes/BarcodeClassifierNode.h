#pragma once

#include "demux/BarcodeClassifierSelector.h"
#include "read_pipeline/base/MessageSink.h"
#include "utils/concurrency/async_task_executor.h"

#include <atomic>
#include <map>
#include <mutex>
#include <string>

namespace dorado {

namespace demux {
struct BarcodingInfo;
}

namespace utils::concurrency {
class MultiQueueThreadPool;
}  // namespace utils::concurrency

class BarcodeClassifierNode : public MessageSink {
public:
    BarcodeClassifierNode(std::shared_ptr<utils::concurrency::MultiQueueThreadPool> thread_pool,
                          utils::concurrency::TaskPriority pipeline_priority);
    BarcodeClassifierNode(int threads);

    ~BarcodeClassifierNode();

    std::string get_name() const override;
    stats::NamedStats sample_stats() const override;
    void terminate(const TerminateOptions&) override;
    void restart() override;

private:
    std::shared_ptr<utils::concurrency::MultiQueueThreadPool> m_thread_pool{};
    utils::concurrency::AsyncTaskExecutor m_task_executor;

    std::atomic<int> m_num_records{0};
    demux::BarcodeClassifierSelector m_barcoder_selector{};

    void input_thread_fn();
    void barcode(BamMessage& read, const demux::BarcodingInfo* barcoding_info);
    void barcode(SimplexRead& read);

    // Track how many reads were classified as each barcode for debugging
    // purposes.
    std::map<std::string, size_t> m_barcode_count;
    mutable std::mutex m_barcode_count_mutex;
};

}  // namespace dorado
