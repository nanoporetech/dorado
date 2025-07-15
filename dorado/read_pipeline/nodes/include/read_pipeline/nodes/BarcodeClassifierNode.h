#pragma once

#include "demux/BarcodeClassifierSelector.h"
#include "read_pipeline/base/MessageSink.h"

#include <atomic>
#include <map>
#include <mutex>
#include <string>

namespace dorado {

namespace demux {
struct BarcodingInfo;
}

class BarcodeClassifierNode : public MessageSink {
public:
    BarcodeClassifierNode(int threads);
    ~BarcodeClassifierNode();

    std::string get_name() const override;
    stats::NamedStats sample_stats() const override;
    void terminate(const TerminateOptions&) override;
    void restart() override;

private:
    std::atomic<int> m_num_records{0};
    demux::BarcodeClassifierSelector m_barcoder_selector{};

    void input_thread_fn();
    void barcode(BamMessage& read, const demux::BarcodingInfo* barcoding_info);
    void barcode(SimplexRead& read);

    // Track how many reads were classified as each barcode for debugging
    // purposes.
    std::map<std::string, std::atomic<size_t>> m_barcode_count;
    std::mutex m_barcode_count_mutex;
};

}  // namespace dorado
