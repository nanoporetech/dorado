#include "demux/adapter_info.h"

namespace {

// This map should contain any primer sets defined in adapter_primer_kits.cpp
// that must be specified in the AdapterInfo object in order to be searched for.
// If one of these entries is specified, then those primers will be searched for
// instead of the standard primers associated with the sequencing kit.
std::map<std::string, dorado::demux::PrimerAux> primer_set_map{
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

std::vector<std::string> extended_primer_names() {
    std::vector<std::string> names;
    for (const auto& item : primer_set_map) {
        names.push_back(item.first);
    }
    return names;
}

}  // namespace dorado::demux
