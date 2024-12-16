#include "AdapterDetectorSelector.h"

#include "AdapterDetector.h"
#include "adapter_info.h"

#include <spdlog/spdlog.h>

#include <cassert>

namespace dorado::demux {

std::shared_ptr<AdapterDetector> AdapterDetectorSelector::get_detector(
        const AdapterInfo& adapter_info) {
    std::string key = "default";
    if (adapter_info.custom_seqs.has_value()) {
        key = adapter_info.custom_seqs.value();
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_detector_lut.count(key)) {
        m_detector_lut.emplace(key, std::make_shared<AdapterDetector>(adapter_info.custom_seqs));
    }
    return m_detector_lut.at(key);
}

}  // namespace dorado::demux
