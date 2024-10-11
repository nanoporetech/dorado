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
private:
    CudaChunkBenchmarks();
    using ChunkTimings = std::unordered_map<int, float>;
    using ModelName = std::string;
    using GPUName = std::string;

    std::mutex m_chunk_benchmarks_mutex;
    std::map<std::pair<GPUName, ModelName>, ChunkTimings> m_chunk_benchmarks;

    // Must be called with m_chunk_benchmarks_mutex already locked.
    std::optional<const ChunkTimings> get_chunk_timings_internal(
            const GPUName& gpu_name,
            const std::string& model_path) const;

public:
    static CudaChunkBenchmarks& instance() {
        static CudaChunkBenchmarks chunk_benchmarks;
        return chunk_benchmarks;
    }

    std::optional<const ChunkTimings> get_chunk_timings(const GPUName& gpu_name,
                                                        const std::string& model_path) const;

    bool add_chunk_timings(const GPUName& gpu_name,
                           const std::string& model_path,
                           const std::vector<std::pair<float, int>>& timings);
};

}  // namespace dorado::basecall
