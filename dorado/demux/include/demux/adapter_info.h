#pragma once

#include <map>
#include <optional>
#include <string>

namespace dorado::demux {

/// This enum allows for specification of special primer detection and handling.
enum class PrimerAux {
    DEFAULT,  ///< Indicates that standard primers for the kit should be used.
    GEN10X,   ///< Applies to SQK-LSK114* kits. Look for 10X primers, and apply special analysis.
    UNKNOWN,  ///< Indicates an unknown (and unsupported) special primer.
};

inline PrimerAux special_primer_by_name(const std::string& name) {
    static std::map<std::string, PrimerAux> name_map{{"", PrimerAux::DEFAULT},
                                                     {"10X", PrimerAux::GEN10X}};

    auto ptr = name_map.find(name);
    if (ptr == name_map.end()) {
        return PrimerAux::UNKNOWN;
    }
    return ptr->second;
}

struct AdapterInfo {
    bool trim_adapters{true};
    bool trim_primers{true};
    bool rna_adapters{false};
    PrimerAux primer_aux{PrimerAux::DEFAULT};
    std::optional<std::string> kit_name = std::nullopt;
    std::optional<std::string> custom_seqs = std::nullopt;
};

}  // namespace dorado::demux
