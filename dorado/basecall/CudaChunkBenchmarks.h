#pragma once

#include <map>
#include <mutex>
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

    mutable std::mutex m_chunk_benchmarks_mutex;
    std::map<std::tuple<GPUName, ModelName, ChunkSize>, ChunkTimings> m_chunk_benchmarks;

    // Must be called with m_chunk_benchmarks_mutex already locked.
    std::optional<const ChunkTimings> get_chunk_timings_internal(const GPUName& gpu_name,
                                                                 const std::string& model_path,
                                                                 ChunkSize chunk_size) const;

public:
    static CudaChunkBenchmarks& instance() {
        static CudaChunkBenchmarks chunk_benchmarks;
        return chunk_benchmarks;
    }

    std::optional<const ChunkTimings> get_chunk_timings(const GPUName& gpu_name,
                                                        const std::string& model_path,
                                                        ChunkSize chunk_size) const;

    bool add_chunk_timings(const GPUName& gpu_name,
                           const std::string& model_path,
                           ChunkSize chunk_size,
                           const std::vector<std::pair<float, int>>& timings);
};

}  // namespace dorado::basecall
