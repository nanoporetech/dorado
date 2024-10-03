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

std::optional<const CudaChunkBenchmarks::ChunkTimings> CudaChunkBenchmarks::get_chunk_timings(
        GPUName gpu_name,
        const std::string& model_path) const {
    // Strip any extra path elements from the model folder name
    ModelName model_name = std::filesystem::path(model_path).filename().string();

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

    if (gpu_name_alias.find(gpu_name) != gpu_name_alias.cend()) {
        gpu_name = gpu_name_alias[gpu_name];
        iter = m_chunk_benchmarks.find({gpu_name, model_name});
        if (iter != m_chunk_benchmarks.cend()) {
            return iter->second;
        }
    }

    return {};
}

bool CudaChunkBenchmarks::add_chunk_timings(const GPUName& gpu_name,
                                            const std::string& model_path,
                                            const std::vector<std::pair<float, int>>& timings) {
    // Strip any extra path elements from the model folder name
    ModelName model_name = std::filesystem::path(model_path).filename().string();

    if (get_chunk_timings(gpu_name, model_name)) {
        return false;
    }

    auto& new_benchmarks = m_chunk_benchmarks[{gpu_name, model_name}];
    for (auto& timing : timings) {
        new_benchmarks[timing.second] = timing.first;
    }

    return true;
}

}  // namespace dorado::basecall
