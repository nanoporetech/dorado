#include "ChunkBenchmarks.h"

namespace dorado::basecall {

ChunkBenchmarks::ChunkBenchmarks() = default;
ChunkBenchmarks::~ChunkBenchmarks() = default;

std::optional<const ChunkBenchmarks::ChunkTimings> ChunkBenchmarks::get_chunk_timings_internal(
        const GPUName& gpu_name,
        const ModelName& model_name) const {
    // Try looking up the specified gpu name directly
    auto iter = m_chunk_benchmarks.find({gpu_name, model_name});
    if (iter != m_chunk_benchmarks.cend()) {
        return iter->second;
    }

    // If the direct lookup fails, try looking up via an alias
    std::map<GPUName, GPUName> gpu_name_alias = {
            {"NVIDIA A100-PCIE-40GB", "NVIDIA A100 80GB PCIe"},
            {"NVIDIA A800 80GB PCIe", "NVIDIA A100 80GB PCIe"},
    };

    auto alias_name = gpu_name_alias.find(gpu_name);
    if (alias_name != gpu_name_alias.cend()) {
        iter = m_chunk_benchmarks.find({alias_name->second, model_name});
        if (iter != m_chunk_benchmarks.cend()) {
            return iter->second;
        }
    }

    return {};
}

std::optional<const ChunkBenchmarks::ChunkTimings> ChunkBenchmarks::get_chunk_timings(
        const GPUName& gpu_name,
        const ModelName& model_name) const {
    std::lock_guard guard(m_chunk_benchmarks_mutex);
    return get_chunk_timings_internal(gpu_name, model_name);
}

bool ChunkBenchmarks::add_chunk_timings(const GPUName& gpu_name,
                                        const ModelName& model_name,
                                        const std::vector<std::pair<float, int>>& timings) {
    std::lock_guard guard(m_chunk_benchmarks_mutex);

    if (get_chunk_timings_internal(gpu_name, model_name)) {
        return false;
    }

    auto& new_benchmarks = m_chunk_benchmarks[{gpu_name, model_name}];
    for (auto& timing : timings) {
        new_benchmarks[timing.second] = timing.first;
    }

    return true;
}

}  // namespace dorado::basecall
