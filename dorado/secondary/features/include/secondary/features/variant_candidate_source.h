#pragma once

#include <string>

namespace dorado::secondary {

enum class VariantCandidateSource {
    NONE,
    FILE,
    COMPUTE,
};

inline std::string variant_region_source_to_string(const VariantCandidateSource hs) {
    switch (hs) {
    case VariantCandidateSource::COMPUTE:
        return "COMPUTE";
    case VariantCandidateSource::FILE:
        return "FILE";
    case VariantCandidateSource::NONE:
        return "NONE";
    }
    throw std::runtime_error{"Unsupported variant candidate source!"};
}

}  // namespace dorado::secondary
