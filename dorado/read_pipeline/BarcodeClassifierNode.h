#pragma once

#include "demux/BarcodeClassifierSelector.h"
#include "read_pipeline/MessageSink.h"
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

class BarcodeClassifierNode : public MessageSink {
public:
    BarcodeClassifierNode(int threads,
                          const std::vector<std::string>& kit_name,
                          bool barcode_both_ends,
                          bool no_trim,
                          BarcodingInfo::FilterSet allowed_barcodes,
                          const std::optional<std::string>& custom_kit,
                          const std::optional<std::string>& custom_seqs);
    BarcodeClassifierNode(int threads);
    ~BarcodeClassifierNode() { stop_input_processing(); }
    std::string get_name() const override { return "BarcodeClassifierNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions&) override { stop_input_processing(); }
    void restart() override {
        start_input_processing(&BarcodeClassifierNode::input_thread_fn, this);
    }

private:
    std::atomic<int> m_num_records{0};
    demux::BarcodeClassifierSelector m_barcoder_selector{};

    void input_thread_fn();
    void barcode(BamPtr& read, const BarcodingInfo* barcoding_info);
    void barcode(SimplexRead& read);

    // Track how many reads were classified as each barcode for debugging
    // purposes.
    std::map<std::string, std::atomic<size_t>> m_barcode_count;
    std::mutex m_barcode_count_mutex;
};

}  // namespace dorado
