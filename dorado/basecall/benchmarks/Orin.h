#pragma once

#include <map>
#include <string>
#include <unordered_map>

namespace dorado::basecall {
void AddOrinBenchmarks(std::map<std::pair<std::string, std::string>,
                                std::unordered_map<int, float>>& chunk_benchmarks);
}  // namespace dorado::basecall
