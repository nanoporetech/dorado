#pragma once

#include <map>
#include <string>
#include <unordered_map>

namespace dorado::basecall {
void AddTesla_V100_PCIE_16GBBenchmarks(std::map<std::pair<std::string, std::string>,
                                                std::unordered_map<int, float>>& chunk_benchmarks);
}  // namespace dorado::basecall
