#include "IndexFileAccess.h"

#include "Minimap2Index.h"

#include <cassert>

namespace dorado::alignment {

std::shared_ptr<Minimap2Index> IndexFileAccess::get_index_impl(
        const std::string& file,
        const Minimap2IndexOptions& options) {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_index_lut[{file, options}];  // allow the creation of a null instance
}

IndexLoadResult IndexFileAccess::load_index(const std::string& file,
                                            const Minimap2Options& options,
                                            int num_threads) {
    auto index = get_index_impl(file, options);
    if (index) {
        return IndexLoadResult::success;
    }

    index = std::make_shared<Minimap2Index>();
    if (!index->initialise(options)) {
        return IndexLoadResult::validation_error;
    }

    auto load_result = index->load(file, num_threads);
    if (load_result != IndexLoadResult::success) {
        return load_result;
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    m_index_lut[{file, options}] = std::move(index);
    return IndexLoadResult::success;
}

std::shared_ptr<const Minimap2Index> IndexFileAccess::get_index(
        const std::string& file,
        const Minimap2IndexOptions& options) {
    auto index = get_index_impl(file, options);
    assert(index && "get_index expects the index to have been loaded");
    return index;
}

}  // namespace dorado::alignment