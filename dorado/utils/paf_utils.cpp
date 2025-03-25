#include "paf_utils.h"

#include "alignment_utils.h"
#include "overlap.h"

#include <spdlog/spdlog.h>

#include <iostream>
#include <ostream>
#include <sstream>

namespace dorado::utils {

PafEntry parse_paf(std::istringstream& ss) {
    PafEntry entry;
    // Read the fields from the stringstream
    ss >> entry.qname >> entry.qlen >> entry.qstart >> entry.qend >> entry.strand >> entry.tname >>
            entry.tlen >> entry.tstart >> entry.tend >> entry.num_residue_matches >>
            entry.alignment_block_length >> entry.mapq;

    // The rest of the line is auxiliary data
    std::getline(ss, entry.aux);

    // Remove the leading tab from aux if it exists
    if (!entry.aux.empty() && entry.aux[0] == '\t') {
        entry.aux.erase(0, 1);
    }

    return entry;
}

PafEntry parse_paf(const std::string& paf_row) {
    std::istringstream ss(paf_row);
    return parse_paf(ss);
}

std::string serialize_paf(const PafEntry& entry) {
    std::ostringstream oss;
    oss << entry.qname << '\t' << entry.qlen << '\t' << entry.qstart << '\t' << entry.qend << '\t'
        << entry.strand << '\t' << entry.tname << '\t' << entry.tlen << '\t' << entry.tstart << '\t'
        << entry.tend << '\t' << entry.num_residue_matches << '\t' << entry.alignment_block_length
        << '\t' << entry.mapq;
    if (!std::empty(entry.aux)) {
        oss << '\t' << entry.aux;
    }
    return oss.str();
}

void serialize_to_paf(std::ostream& os,
                      const std::string_view qname,
                      const std::string_view tname,
                      const Overlap& ovl,
                      const int num_residue_matches,
                      const int alignment_block_length,
                      const int mapq,
                      const std::vector<CigarOp>& cigar) {
    os << qname << '\t' << ovl.qlen << '\t' << ovl.qstart << '\t' << ovl.qend << '\t'
       << (ovl.fwd ? '+' : '-') << '\t' << tname << '\t' << ovl.tlen << '\t' << ovl.tstart << '\t'
       << ovl.tend << '\t' << num_residue_matches << '\t' << alignment_block_length << '\t' << mapq;
    if (!std::empty(cigar)) {
        os << "\tcg:Z:" << cigar;
    }
}

std::string_view paf_aux_get(const PafEntry& paf_entry, const char tag[2], const char type) {
    const std::string t = std::string(tag) + ":" + std::string(1, type) + ":";
    const std::string_view aux(paf_entry.aux);
    size_t pos = aux.find(t.c_str());
    if (pos == std::string::npos) {
        return {};
    }
    pos += 5;
    const size_t end = aux.find('\t', pos);
    if (end == std::string::npos) {
        return aux.substr(pos);
    }
    assert(end > pos);
    return aux.substr(pos, end - pos);
}

std::ostream& operator<<(std::ostream& os, const Overlap& overlap) {
    serialize_to_paf(os, "query", "target", overlap, 0, 0, 0, {});
    return os;
}

}  // namespace dorado::utils
