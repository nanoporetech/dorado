#pragma once

#include <optional>
#include <string>
#include <vector>

namespace dorado::demux {

/// This enum allows for specification of extended primer detection and handling.
enum class PrimerAux {
    DEFAULT,  ///< Indicates that standard primers for the kit should be used.
    GEN10X,   ///< Applies to SQK-LSK114* kits. Look for 10X primers, and apply special analysis.
    UNKNOWN,  ///< Indicates an unknown (and unsupported) extended primer.
};

/// Get the primer sequences for the specified primer set.
PrimerAux extended_primers_by_name(const std::string& name);

/// Get the names of all supported extended primer sets.
const std::vector<std::string>& extended_primer_names();

struct AdapterInfo {
    bool trim_adapters{true};
    bool trim_primers{true};
    bool rna_adapters{false};
    PrimerAux primer_aux{PrimerAux::DEFAULT};
    std::optional<std::string> kit_name = std::nullopt;
    std::optional<std::string> custom_seqs = std::nullopt;

    bool set_primer_sequences(const std::string& primer_sequences);
};

}  // namespace dorado::demux
