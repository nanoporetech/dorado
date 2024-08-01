#pragma once

#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace dorado::basecall {

class CudaChunkBenchmarks final {
private:
    CudaChunkBenchmarks();
    using ChunkTimings = std::map<int, float>;
    using ModelName = std::string;
    using GPUName = std::string;
    using ChunkSize = int;
    std::map<std::tuple<GPUName, ModelName, ChunkSize>, ChunkTimings> m_chunk_benchmarks;

public:
    static CudaChunkBenchmarks& instance() {
        static CudaChunkBenchmarks chunk_benchmarks;
        return chunk_benchmarks;
    }

    std::optional<const ChunkTimings> get_chunk_timings(GPUName gpu_name,
                                                        const ModelName& model_name,
                                                        ChunkSize chunk_size) const;

    bool add_chunk_timings(GPUName gpu_name,
                           const ModelName& model_name,
                           ChunkSize chunk_size,
                           std::vector<std::pair<float, int>> timings);
};

}  // namespace dorado::basecall
