#pragma once

#include <map>
#include <string>
#include <unordered_map>

namespace dorado::basecall {
void AddNVIDIA_RTX_PRO_6000_Blackwell_Max_Q_Workstation_EditionBenchmarks(
        std::map<std::pair<std::string, std::string>, std::unordered_map<int, float>>&
                chunk_benchmarks);
}  // namespace dorado::basecall
