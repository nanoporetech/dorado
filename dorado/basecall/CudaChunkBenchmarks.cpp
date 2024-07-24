#include "CudaChunkBenchmarks.h"

#include "benchmarks/NVIDIA_A100_80GB_PCIe.h"
#include "benchmarks/NVIDIA_RTX_A6000.h"
#include "benchmarks/Quadro_GV100.h"
#include "benchmarks/Tesla_V100-PCIE-16GB.h"

namespace dorado::basecall {

CudaChunkBenchmarks::CudaChunkBenchmarks() {
    AddNVIDIA_A100_80GB_PCIeBenchmarks(m_chunk_benchmarks);
    AddNVIDIA_RTX_A6000Benchmarks(m_chunk_benchmarks);
    AddQuadro_GV100Benchmarks(m_chunk_benchmarks);
    AddTesla_V100_PCIE_16GBBenchmarks(m_chunk_benchmarks);
}

}  // namespace dorado::basecall
