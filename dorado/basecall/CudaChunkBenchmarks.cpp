#include "CudaChunkBenchmarks.h"

#include "Quadro_GV100.h"

namespace dorado::basecall {

CudaChunkBenchmarks::CudaChunkBenchmarks() { AddQuadro_GV100Benchmarks(m_chunk_benchmarks); }

}  // namespace dorado::basecall
