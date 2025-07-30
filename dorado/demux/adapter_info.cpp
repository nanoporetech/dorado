#include "demux/adapter_info.h"

#include "utils/string_utils.h"

#include <spdlog/spdlog.h>

#include <map>

namespace {

// This map should contain any primer sets defined in adapter_primer_kits.cpp
// that must be specified in the AdapterInfo object in order to be searched for.
// If one of these entries is specified, then those primers will be searched for
// instead of the standard primers associated with the sequencing kit.
const std::map<std::string, dorado::demux::PrimerAux> primer_set_map{
        {"10X_Genomics", dorado::demux::PrimerAux::GEN10X}};

}  // namespace

namespace dorado::demux {

PrimerAux extended_primers_by_name(const std::string& name) {
    if (name.empty()) {
        return PrimerAux::DEFAULT;
    }
    auto ptr = primer_set_map.find(name);
    if (ptr == primer_set_map.end()) {
        return PrimerAux::UNKNOWN;
    }
    return ptr->second;
}

const std::vector<std::string>& extended_primer_names() {
    static std::vector<std::string> names;
    if (names.empty()) {
        names.reserve(primer_set_map.size());
        for (const auto& item : primer_set_map) {
            names.push_back(item.first);
        }
    }
    return names;
}

bool AdapterInfo::set_primer_sequences(const std::string& primer_sequences) {
    if (utils::ends_with(primer_sequences, ".fa") || utils::ends_with(primer_sequences, ".fasta")) {
        custom_seqs = primer_sequences;
        return true;
    }
    auto aux_code_entry = primer_set_map.find(primer_sequences);
    if (aux_code_entry == primer_set_map.end()) {
        const std::string extended_primer_codes = utils::join(demux::extended_primer_names(), ", ");
        spdlog::error(
                "Error: Invalid value {} for --primer_sequences option. This must be either the "
                "full path of a fasta file, or a supported 3rd-party primer set code. Supported "
                "codes are: {}",
                primer_sequences, extended_primer_codes);
        return false;
    }
    primer_aux = aux_code_entry->second;
    return true;
}

}  // namespace dorado::demux
