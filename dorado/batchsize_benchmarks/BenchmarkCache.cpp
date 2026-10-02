#include "BenchmarkCache.h"

#include "compiled_timings.h"
#include "csv_helpers.h"
#include "utils/string_utils.h"

#include <spdlog/spdlog.h>

#include <fstream>
#include <optional>
#include <string>

namespace dorado::batchsize_benchmarks {

namespace {

std::optional<std::string_view> get_gpu_name_alias(std::string_view gpu_name) {
    constexpr std::pair<std::string_view, std::string_view> gpu_name_alias[]{
            {"NVIDIA A100-PCIE-40GB", "NVIDIA A100 80GB PCIe"},
            {"NVIDIA A800 80GB PCIe", "NVIDIA A100 80GB PCIe"},
    };
    for (const auto &alias : gpu_name_alias) {
        if (alias.first == gpu_name) {
            return alias.second;
        }
    }
    return std::nullopt;
}

std::span<const SpeedEntry> get_from_compiled_cache(std::string_view gpu_name,
                                                    std::string_view model_name) {
    auto all_gpus = compiled_cache::get();

    // Find the GPU.
    const auto gpu_alias = get_gpu_name_alias(gpu_name);
    auto is_gpu = [gpu_name, gpu_alias](const compiled_cache::GPUModelTimings &gpu) {
        return gpu.gpu_name == gpu_name || gpu.gpu_name == gpu_alias;
    };
    auto gpu_it = std::find_if(all_gpus.begin(), all_gpus.end(), is_gpu);
    if (gpu_it == all_gpus.end()) {
        return {};
    }

    // Find the model.
    auto all_models = gpu_it->models;
    auto is_model = [model_name](const compiled_cache::ModelTimings &model) {
        return model.model_name == model_name;
    };
    auto model_it = std::find_if(all_models.begin(), all_models.end(), is_model);
    if (model_it == all_models.end()) {
        return {};
    }

    return model_it->entries;
}

}  // namespace

void BenchmarkCache::CacheProxy::add_timings(std::string gpu_name,
                                             std::string model_name,
                                             std::vector<SpeedEntry> entries) {
    if (utils::contains(gpu_name, ",") || utils::contains(model_name, ",")) {
        // A comma will break the rudimentary CSV reader, but we still allow them to be cached.
        DORADO_LOG(warn,
                   "GPU or model name has unexpected character. Exported benchmarks won't be "
                   "loadable");
    }

    // Entries need to be sorted for lookup to work.
    // Note that we intentionally keep all of the timings in the runtime cache so that
    // they all get written out.
    std::sort(entries.begin(), entries.end(), [](const SpeedEntry &lhs, const SpeedEntry &rhs) {
        return lhs.batch_size < rhs.batch_size;
    });

    auto key = Key(std::move(gpu_name), std::move(model_name));
    auto &speeds = m_cache.m_runtime_cache[std::move(key)];
    speeds.swap(entries);
}

std::span<const SpeedEntry> BenchmarkCache::CacheProxy::get_timings(
        const std::string_view gpu_name,
        const std::string_view model_name) const {
#if (defined(_GLIBCXX_RELEASE) && _GLIBCXX_RELEASE < 14) || (defined(_MSC_VER) && _MSC_VER < 1937)
    // There's a bug in libstdc++'s std::pair comparator before GCC 14, and MSVC's STL before 19.37.
    auto key = std::make_pair(std::string(gpu_name), std::string(model_name));
#else
    auto key = std::make_pair(gpu_name, model_name);
#endif

    // Check the runtime cache first.
    const auto &runtime_cache = m_cache.m_runtime_cache;
    auto runtime_it = runtime_cache.find(key);
    if (runtime_it == runtime_cache.end()) {
        // If we can't find it, check it again with the alias.
        if (const auto alias_name = get_gpu_name_alias(gpu_name); alias_name.has_value()) {
            key.first = alias_name.value();
            runtime_it = runtime_cache.find(key);
        }
    }
    if (runtime_it != runtime_cache.end()) {
        return runtime_it->second;
    }

    // Fall back to the compiled cache.
    return get_from_compiled_cache(gpu_name, model_name);
}

bool BenchmarkCache::CacheProxy::load_from_file(const std::filesystem::path &path) {
    std::ifstream file(path);
    if (!file) {
        return false;
    }

    // Read the new entries in.
    std::map<Key, std::vector<SpeedEntry>, std::less<>> new_entries;
    auto process_line = [&](std::string_view gpu_name, std::string_view model_name,
                            const SpeedEntry &entry) {
        const auto key = Key(gpu_name, model_name);
        new_entries[key].push_back(entry);
    };
    csv_read_lines(file, process_line);

    // Update the cache.
    for (auto &[key, entries] : new_entries) {
        const auto &[gpu_name, model_name] = key;
        add_timings(gpu_name, model_name, std::move(entries));
    }

    return true;
}

bool BenchmarkCache::CacheProxy::export_to_file(const std::filesystem::path &path) const {
    std::ofstream file(path);
    if (!file) {
        return false;
    }

    // Write the runtime cache out as a CSV.
    // We assume that the compiled cache has been stored somewhere so doesn't need writing out again.
    for (const auto &[key, entries] : m_cache.m_runtime_cache) {
        const auto &[gpu_name, model_name] = key;
        csv_write_entries(file, gpu_name, model_name, entries);
    }

    return true;
}

BenchmarkCache::BenchmarkCache() = default;

BenchmarkCache::~BenchmarkCache() = default;

BenchmarkCache BenchmarkCache::s_cache;

}  // namespace dorado::batchsize_benchmarks
