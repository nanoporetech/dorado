#pragma once

#include <map>
#include <string>
#include <tuple>

namespace dorado::basecall {
void AddNVIDIA_RTX_A6000Benchmarks(
        std::map<std::tuple<std::string, std::string>, std::map<int, float>>& chunk_benchmarks);
}  // namespace dorado::basecall
