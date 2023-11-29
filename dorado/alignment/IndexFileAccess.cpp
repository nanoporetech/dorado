#include "IndexFileAccess.h"

#include "Minimap2Index.h"

#include <cassert>

namespace dorado::alignment {

const Minimap2Index* IndexFileAccess::get_compatible_index(
        const std::string& file,
        const Minimap2IndexOptions& indexing_options) {
    auto& compatible_indices = m_index_lut[{file, indexing_options}];
    if (compatible_indices.empty()) {
        return nullptr;
    }
    return compatible_indices.begin()->second.get();
}

std::shared_ptr<Minimap2Index> IndexFileAccess::get_exact_index(const std::string& file,
                                                                const Minimap2Options& options) {
    auto& compatible_indices = m_index_lut[{file, options}];
    if (compatible_indices.count(options) == 0) {
        return nullptr;
    }
    return compatible_indices.at(options);
}

bool IndexFileAccess::is_index_loaded_impl(const std::string& file,
                                           const Minimap2Options& options) const {
    const auto compatible_indices = m_index_lut.find({file, options});
    if (compatible_indices == m_index_lut.end()) {
        return false;
    }
    return compatible_indices->second.count(options) > 0;
}

bool IndexFileAccess::try_load_compatible_index(const std::string& file,
                                                const Minimap2Options& options) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (is_index_loaded_impl(file, options)) {
        return true;
    }

    auto compatible_index = get_compatible_index(file, options);
    if (!compatible_index) {
        return false;
    }

    auto new_index = compatible_index->create_compatible_index(options);
    if (!new_index) {
        return false;
    }
    m_index_lut[{file, options}][options] = std::move(new_index);
    return true;
}

IndexLoadResult IndexFileAccess::load_index(const std::string& file,
                                            const Minimap2Options& options,
                                            int num_threads) {
    if (try_load_compatible_index(file, options)) {
        return IndexLoadResult::success;
    }

    auto new_index = std::make_shared<Minimap2Index>();
    if (!new_index->initialise(options)) {
        return IndexLoadResult::validation_error;
    }

    auto load_result = new_index->load(file, num_threads);
    if (load_result != IndexLoadResult::success) {
        return load_result;
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    m_index_lut[{file, options}][options] = std::move(new_index);
    return IndexLoadResult::success;
}

std::shared_ptr<const Minimap2Index> IndexFileAccess::get_index(const std::string& file,
                                                                const Minimap2Options& options) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto index = get_exact_index(file, options);
    assert(index && "get_index expects the index to have been loaded");
    return index;
}

bool IndexFileAccess::is_index_loaded(const std::string& file,
                                      const Minimap2Options& options) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return is_index_loaded_impl(file, options);
}

void IndexFileAccess::unload_index(const std::string& file,
                                   const Minimap2IndexOptions& index_options) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_index_lut[{file, index_options}] = {};
}

bool validate_options(const Minimap2Options& options) {
    Minimap2Index index{};
    return index.initialise(options);
}

}  // namespace dorado::alignment