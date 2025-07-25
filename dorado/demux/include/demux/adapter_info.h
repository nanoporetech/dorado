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
    // This map should contain any special primers defined in adapter_primer_kits.cpp
    // that must be specified in the AdapterInfo object in order to be searched for.
    // If one of these entries is specified, then those primers will be searched for
    // instead of the standard primers associated with the sequencing kit.
    static std::map<std::string, PrimerAux> name_map{{"10X_Genomics", PrimerAux::GEN10X}};

    if (name.empty()) {
        return PrimerAux::DEFAULT;
    }
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
