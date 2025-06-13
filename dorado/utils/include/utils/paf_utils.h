#pragma once

#include "cigar.h"

#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

namespace dorado::utils {

struct Overlap;

struct PafEntry {
    std::string qname{"*"};
    int qlen = 0;
    int qstart = 0;
    int qend = 0;
    char strand{'*'};
    std::string tname{"*"};
    int tlen = 0;
    int tstart = 0;
    int tend = 0;
    int num_residue_matches = 0;
    int alignment_block_length = 0;
    int mapq = 0;
    std::string aux;

    void add_aux_tag(const char tag[2], char type, const std::string& data) {
        if (!aux.empty()) {
            aux += '\t';
        }
        std::string t = std::string(tag) + ":" + std::string(1, type) + ":";
        aux.append(t);
        aux.append(data);
    }
};

PafEntry parse_paf(const std::string& paf_row);
PafEntry parse_paf(std::istringstream& paf_row);

std::string serialize_paf(const PafEntry& paf_entry);

void serialize_to_paf(std::ostream& os,
                      std::string_view qname,
                      std::string_view tname,
                      const Overlap& overlap,
                      int num_residue_matches,
                      int alignment_block_length,
                      int mapq,
                      const std::vector<CigarOp>& cigar);

std::ostream& operator<<(std::ostream& os, const Overlap& overlap);

std::string_view paf_aux_get(const PafEntry& paf_entry, const char tag[2], char type);

}  // namespace dorado::utils
