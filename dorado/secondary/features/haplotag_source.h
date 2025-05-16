#pragma once

#include <string>

namespace dorado::secondary {

enum class HaplotagSource {
    COMPUTE,
    BIN_FILE,
    BAM_HAP_TAG,
    UNPHASED,
};

inline std::string haplotag_source_to_string(const HaplotagSource hs) {
    switch (hs) {
    case HaplotagSource::COMPUTE:
        return "COMPUTE";
    case HaplotagSource::BIN_FILE:
        return "BIN_FILE";
    case HaplotagSource::BAM_HAP_TAG:
        return "BAM_HAP_TAG";
    case HaplotagSource::UNPHASED:
        return "UNPHASED";
    }
    throw std::runtime_error{"Unsupported haplotag source!"};
}

}  // namespace dorado::secondary
