#pragma once

#include "hts_utils/hts_types.h"

#include <array>
#include <filesystem>
#include <string>
#include <unordered_map>

struct sam_hdr_t;

namespace dorado {
class HtsData;

using AlignmentCounts = std::unordered_map<std::string, std::array<int, 3>>;

class ReadInitialiser {
public:
    ReadInitialiser(sam_hdr_t* hdr, AlignmentCounts aln_counts);
    void update_read_attributes(HtsData& data) const;
    void update_barcoding_fields(HtsData& data) const;
    void update_alignment_fields(HtsData& data) const;

private:
    sam_hdr_t* m_header;
    AlignmentCounts m_alignment_counts;
    std::unordered_map<std::string, dorado::ReadGroup> m_read_groups;
    int m_minimum_qscore;
};

void update_alignment_counts(const std::filesystem::path& path, AlignmentCounts& alignment_counts);

}  // namespace dorado
