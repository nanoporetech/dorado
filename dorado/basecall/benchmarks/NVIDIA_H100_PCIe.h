#pragma once

#include <map>
#include <string>
#include <tuple>

namespace dorado::basecall {
void AddNVIDIA_H100_PCIeBenchmarks(std::map<std::tuple<std::string, std::string, int>,
                                            std::map<int, float>>& chunk_benchmarks);
}  // namespace dorado::basecall
