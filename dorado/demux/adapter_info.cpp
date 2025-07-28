#include "demux/adapter_info.h"

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

}  // namespace dorado::demux
