#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace kadayashi {

std::vector<std::string> split_deli_line(const std::string &line, const char delimeter);
void write_binary_given_tsv(const std::filesystem::path &fn_tsv,
                            const std::filesystem::path &fn_bin);
std::unordered_map<std::string, int> query_bin_file_get_qname2hp(
        const std::filesystem::path &fn_bin,
        const std::string &chrom,
        const uint32_t ref_start,
        const uint32_t ref_end);
}  // namespace kadayashi
