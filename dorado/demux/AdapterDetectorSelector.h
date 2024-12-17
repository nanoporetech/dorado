#pragma once
#include "utils/types.h"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace dorado::demux {

class AdapterDetector;
struct AdapterInfo;

class AdapterDetectorSelector final {
    std::mutex m_mutex{};
    std::unordered_map<std::string, std::shared_ptr<AdapterDetector>> m_detector_lut{};

public:
    std::shared_ptr<AdapterDetector> get_detector(const AdapterInfo& adapter_info);
};

}  // namespace dorado::demux
