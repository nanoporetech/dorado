#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_set>

namespace dorado::secondary {

struct BamInfo {
    bool uses_dorado_aligner = false;
    bool has_dwells = false;
    std::unordered_set<std::string> read_groups;
    std::unordered_set<std::string> basecaller_models;
};

/**
 * \brief Opens the input BAM file and summarizes some information needed at runtime.
 * \param in_aln_bam_fn Path to the input BAM file.
 * \param cli_read_group If not empty, only this read group will be loaded from the BAM header.
 */
BamInfo analyze_bam(const std::filesystem::path& in_aln_bam_fn, const std::string& cli_read_group);

void check_read_groups(const BamInfo& bam_info, const std::string& cli_read_group);

}  // namespace dorado::secondary
