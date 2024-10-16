#include "CudaChunkBenchmarks.h"

#include "benchmarks/NVIDIA_A100_80GB_PCIe.h"
#include "benchmarks/NVIDIA_H100_PCIe.h"
#include "benchmarks/NVIDIA_RTX_A6000.h"
#include "benchmarks/Quadro_GV100.h"
#include "benchmarks/Tesla_V100-PCIE-16GB.h"

#include <filesystem>

namespace dorado::basecall {

CudaChunkBenchmarks::CudaChunkBenchmarks() {
    AddNVIDIA_A100_80GB_PCIeBenchmarks(m_chunk_benchmarks);
    AddNVIDIA_H100_PCIeBenchmarks(m_chunk_benchmarks);
    AddNVIDIA_RTX_A6000Benchmarks(m_chunk_benchmarks);
    AddQuadro_GV100Benchmarks(m_chunk_benchmarks);
    AddTesla_V100_PCIE_16GBBenchmarks(m_chunk_benchmarks);
}

std::optional<const CudaChunkBenchmarks::ChunkTimings>
CudaChunkBenchmarks::get_chunk_timings_internal(const GPUName& gpu_name,
                                                const std::string& model_path,
                                                ChunkSize chunk_size) const {
    // Strip any extra path elements from the model folder name
    ModelName model_name = std::filesystem::path(model_path).filename().string();

    // Try looking up the specified gpu name directly
    auto iter = m_chunk_benchmarks.find({gpu_name, model_name, chunk_size});
    if (iter != m_chunk_benchmarks.end()) {
        return iter->second;
    }

    // If the direct lookup fails, try looking up via an alias
    std::map<GPUName, GPUName> gpu_name_alias = {
            {"NVIDIA A100-PCIE-40GB", "NVIDIA A100 80GB PCIe"},
            {"NVIDIA A800 80GB PCIe", "NVIDIA A100 80GB PCIe"},
    };

    auto alias_name = gpu_name_alias.find(gpu_name);
    if (alias_name != gpu_name_alias.cend()) {
        iter = m_chunk_benchmarks.find({alias_name->second, model_name, chunk_size});
        if (iter != m_chunk_benchmarks.cend()) {
            return iter->second;
        }
    }

    return {};
}

std::optional<const CudaChunkBenchmarks::ChunkTimings> CudaChunkBenchmarks::get_chunk_timings(
        const GPUName& gpu_name,
        const std::string& model_path,
        ChunkSize chunk_size) const {
    std::lock_guard guard(m_chunk_benchmarks_mutex);
    return get_chunk_timings_internal(gpu_name, model_path, chunk_size);
}

bool CudaChunkBenchmarks::add_chunk_timings(const GPUName& gpu_name,
                                            const std::string& model_path,
                                            ChunkSize chunk_size,
                                            const std::vector<std::pair<float, int>>& timings) {
    std::lock_guard guard(m_chunk_benchmarks_mutex);

    // Strip any extra path elements from the model folder name
    ModelName model_name = std::filesystem::path(model_path).filename().string();

    if (get_chunk_timings_internal(gpu_name, model_name, chunk_size)) {
        return false;
    }

    auto& new_benchmarks = m_chunk_benchmarks[{gpu_name, model_name, chunk_size}];
    for (auto& timing : timings) {
        new_benchmarks[timing.second] = timing.first;
    }

    return true;
}

}  // namespace dorado::basecall
