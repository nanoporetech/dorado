#pragma once

#include "MessageSink.h"
#include "demux/BarcodeClassifierSelector.h"
#include "utils/stats.h"
#include "utils/types.h"

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace dorado {

namespace demux {
struct BarcodingInfo;
}

class BarcodeClassifierNode : public MessageSink {
public:
    BarcodeClassifierNode(int threads);
    ~BarcodeClassifierNode() { stop_input_processing(); }
    std::string get_name() const override { return "BarcodeClassifierNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions&) override { stop_input_processing(); }
    void restart() override {
        start_input_processing([this] { input_thread_fn(); }, "brcd_classifier");
    }

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
