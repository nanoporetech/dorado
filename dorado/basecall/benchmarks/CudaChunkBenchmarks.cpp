#include "CudaChunkBenchmarks.h"

#include "NVIDIA_A100_80GB_PCIe.h"
#include "NVIDIA_H100_NVL.h"
#include "NVIDIA_RTX_A6000.h"
#include "Orin.h"
#include "Quadro_GV100.h"
#include "Tesla_V100-PCIE-16GB.h"

namespace dorado::basecall {

CudaChunkBenchmarks::CudaChunkBenchmarks() {
    AddNVIDIA_A100_80GB_PCIeBenchmarks(m_chunk_benchmarks);
    AddNVIDIA_H100_NVLBenchmarks(m_chunk_benchmarks);
    AddNVIDIA_RTX_A6000Benchmarks(m_chunk_benchmarks);
    AddOrinBenchmarks(m_chunk_benchmarks);
    AddQuadro_GV100Benchmarks(m_chunk_benchmarks);
    AddTesla_V100_PCIE_16GBBenchmarks(m_chunk_benchmarks);
}

std::optional<const CudaChunkBenchmarks::ChunkTimings>
CudaChunkBenchmarks::get_chunk_timings_internal(const GPUName& gpu_name,
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
            {"NVIDIA H100 PCIe", "NVIDIA H100 NVL"},
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

std::optional<const CudaChunkBenchmarks::ChunkTimings> CudaChunkBenchmarks::get_chunk_timings(
        const GPUName& gpu_name,
        const ModelName& model_name) const {
    std::lock_guard guard(m_chunk_benchmarks_mutex);
    return get_chunk_timings_internal(gpu_name, model_name);
}

bool CudaChunkBenchmarks::add_chunk_timings(const GPUName& gpu_name,
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
