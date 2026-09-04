#pragma once

#include "SpeedEntry.h"

#include <filesystem>
#include <map>
#include <mutex>
#include <span>
#include <string>
#include <vector>

namespace dorado::batchsize_benchmarks {

class BenchmarkCache final {
public:
    // Perform an action with the cache lock held.
    // Functor will be passed a CacheProxy.
    template <typename Functor>
    static auto with_lock(Functor &&func) {
        std::lock_guard guard(s_cache.m_mutex);
        CacheProxy proxy(s_cache);
        return std::forward<Functor>(func)(proxy);
    }

    class CacheProxy final {
    public:
        explicit CacheProxy(BenchmarkCache &cache) : m_cache(cache) {}

        // Add new timings to the cache.
        // This will replace existing ones with the same gpu+model.
        void add_timings(std::string gpu_name,
                         std::string model_name,
                         std::vector<SpeedEntry> entries);

        // Get the timings for a given config.
        // Returns an empty span if they aren't found.
        // Note: the returned timings are only valid while the lock is held.
        std::span<const SpeedEntry> get_timings(std::string_view gpu_name,
                                                std::string_view model_name) const;

        // Adds new timing data to the cache, as if calling add_timings().
        bool load_from_file(const std::filesystem::path &path);

        // Export the runtime cache to a file.
        bool export_to_file(const std::filesystem::path &path) const;

    private:
        BenchmarkCache &m_cache;
    };

private:
    BenchmarkCache();
    ~BenchmarkCache();
    BenchmarkCache(const BenchmarkCache &) = delete;
    BenchmarkCache &operator=(const BenchmarkCache &) = delete;
    BenchmarkCache(BenchmarkCache &&) = delete;
    BenchmarkCache &operator=(BenchmarkCache &&) = delete;

private:
    std::mutex m_mutex;
    using GPUName = std::string;
    using ModelName = std::string;
    using Key = std::pair<GPUName, ModelName>;
    std::map<Key, std::vector<SpeedEntry>, std::less<>> m_runtime_cache;

    static BenchmarkCache s_cache;
};

}  // namespace dorado::batchsize_benchmarks
