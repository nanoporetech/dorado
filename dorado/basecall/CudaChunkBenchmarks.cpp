#include "CudaChunkBenchmarks.h"

#include "benchmarks/NVIDIA_A100_80GB_PCIe.h"
#include "benchmarks/NVIDIA_H100_PCIe.h"
#include "benchmarks/NVIDIA_RTX_A6000.h"
#include "benchmarks/Quadro_GV100.h"
#include "benchmarks/Tesla_V100-PCIE-16GB.h"

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
        const ModelName& model_name,
        ChunkSize chunk_size) const {
    std::map<GPUName, GPUName> gpu_name_alias = {
            {"NVIDIA A100-PCIE-40GB", "NVIDIA A100 80GB PCIe"},
            {"NVIDIA A800 80GB PCIe", "NVIDIA A100 80GB PCIe"},
    };

    if (gpu_name_alias.find(gpu_name) != gpu_name_alias.end()) {
        gpu_name = gpu_name_alias[gpu_name];
    }

    if (m_chunk_benchmarks.find({gpu_name, model_name, chunk_size}) != m_chunk_benchmarks.end()) {
        return m_chunk_benchmarks.at({gpu_name, model_name, chunk_size});
    }
    return {};
}

bool CudaChunkBenchmarks::add_chunk_timings(const GPUName& gpu_name,
                                            const ModelName& model_name,
                                            ChunkSize chunk_size,
                                            const std::vector<std::pair<float, int>>& timings) {
    if (get_chunk_timings(gpu_name, model_name, chunk_size)) {
        return false;
    }

    auto& new_benchmarks = m_chunk_benchmarks[{gpu_name, model_name, chunk_size}];
    for (auto& timing : timings) {
        new_benchmarks[timing.second] = timing.first;
    }

    return true;
}

}  // namespace dorado::basecall
