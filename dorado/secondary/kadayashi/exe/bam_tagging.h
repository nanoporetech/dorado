#pragma once

#include "secondary/kadayashi/types.h"

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace kadayashi {

int write_haptagged_bam_given_hashtable_and_itvl(
        const std::filesystem::path &fn_bam,
        const char *itvl,
        const std::filesystem::path &fn_out,
        const std::unordered_map<std::string, int> &qname2hp,
        const int n_threads);

int write_haptagged_bam_given_bin_and_itvl(const std::filesystem::path &fn_bam,
                                           const std::filesystem::path &fn_bin,
                                           const char *itvl,
                                           const std::filesystem::path &fn_out,
                                           const int n_threads,
                                           const int n_bam_threads);

int write_haptagged_bam_given_hashtable_and_multiple_itvls(
        const std::filesystem::path &fn_bam,
        const std::vector<std::string> &query_regions,
        const std::filesystem::path &fn_out,
        const std::unordered_map<std::string, int> &qname2hp,
        const int n_threads);

}  // namespace kadayashi
