#pragma once

#include <map>
#include <string>
#include <tuple>

namespace dorado::basecall {
void AddTesla_V100_PCIE_16GBBenchmarks(
        std::map<std::tuple<std::string, std::string>, std::map<int, float>>& chunk_benchmarks);
}  // namespace dorado::basecall
