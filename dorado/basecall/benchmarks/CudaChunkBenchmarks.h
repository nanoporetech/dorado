#pragma once

#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dorado::basecall {

class CudaChunkBenchmarks final {
public:
    using ChunkTimings = std::unordered_map<int, float>;
    using ModelName = std::string;
    using GPUName = std::string;

public:
    static CudaChunkBenchmarks& instance() {
        static CudaChunkBenchmarks chunk_benchmarks;
        return chunk_benchmarks;
    }

    std::optional<const ChunkTimings> get_chunk_timings(const GPUName& gpu_name,
                                                        const ModelName& model_name) const;

    bool add_chunk_timings(const GPUName& gpu_name,
                           const ModelName& model_name,
                           const std::vector<std::pair<float, int>>& timings);

private:
    CudaChunkBenchmarks();
    CudaChunkBenchmarks(const CudaChunkBenchmarks&) = delete;
    CudaChunkBenchmarks& operator=(const CudaChunkBenchmarks&) = delete;

    // Must be called with m_chunk_benchmarks_mutex already locked.
    std::optional<const ChunkTimings> get_chunk_timings_internal(const GPUName& gpu_name,
                                                                 const ModelName& model_name) const;

private:
    mutable std::mutex m_chunk_benchmarks_mutex;
    std::map<std::pair<GPUName, ModelName>, ChunkTimings> m_chunk_benchmarks;
};

}  // namespace dorado::basecall
