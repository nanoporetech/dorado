#include "IndexFileAccess.h"

#include "Minimap2Index.h"

#include <cassert>
#include <sstream>

namespace dorado::alignment {

const Minimap2Index* IndexFileAccess::get_compatible_index(
        const std::string& index_file,
        const Minimap2IndexOptions& indexing_options) {
    auto& compatible_indices = m_index_lut[{index_file, indexing_options}];
    if (compatible_indices.empty()) {
        return nullptr;
    }
    return compatible_indices.begin()->second.get();
}

std::shared_ptr<Minimap2Index> IndexFileAccess::get_exact_index_impl(
        const std::string& index_file,
        const Minimap2Options& options) const {
    auto compatible_indices = m_index_lut.find({index_file, options});
    if (compatible_indices == m_index_lut.end()) {
        return nullptr;
    }
    auto exact_index = compatible_indices->second.find(options);
    return exact_index == compatible_indices->second.end() ? nullptr : exact_index->second;
}

std::shared_ptr<Minimap2Index> IndexFileAccess::get_exact_index(
        const std::string& index_file,
        const Minimap2Options& options) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto exact_index = get_exact_index_impl(index_file, options);
    assert(exact_index && "Cannot access an index which has not been loaded");
    return exact_index;
}

bool IndexFileAccess::is_index_loaded_impl(const std::string& index_file,
                                           const Minimap2Options& options) const {
    const auto compatible_indices = m_index_lut.find({index_file, options});
    if (compatible_indices == m_index_lut.end()) {
        return false;
    }
    return compatible_indices->second.count(options) > 0;
}

std::shared_ptr<Minimap2Index> IndexFileAccess::get_or_load_compatible_index(
        const std::string& index_file,
        const Minimap2Options& options) {
    auto index = get_exact_index_impl(index_file, options);
    if (index) {
        return index;
    }
    auto compatible_index = get_compatible_index(index_file, options);
    if (!compatible_index) {
        return nullptr;
    }

    auto new_index = compatible_index->create_compatible_index(options);
    if (!new_index) {
        return nullptr;
    }
    m_index_lut[{index_file, options}][options] = new_index;
    return new_index;
}

bool IndexFileAccess::try_load_compatible_index(const std::string& index_file,
                                                const Minimap2Options& options) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto index = get_or_load_compatible_index(index_file, options);
    return index != nullptr;
}

IndexLoadResult IndexFileAccess::load_index(const std::string& index_file,
                                            const Minimap2Options& options,
                                            int num_threads) {
    if (try_load_compatible_index(index_file, options)) {
        return IndexLoadResult::success;
    }

    auto new_index = std::make_shared<Minimap2Index>();
    if (!new_index->initialise(options)) {
        return IndexLoadResult::validation_error;
    }

    auto load_result = new_index->load(index_file, num_threads, false);
    if (load_result != IndexLoadResult::success) {
        return load_result;
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    m_index_lut[{index_file, options}][options] = std::move(new_index);
    return IndexLoadResult::success;
}

std::shared_ptr<const Minimap2Index> IndexFileAccess::get_index(const std::string& index_file,
                                                                const Minimap2Options& options) {
    std::lock_guard<std::mutex> lock(m_mutex);
    // N.B. Although the index file for a client must be loaded before reads are added to the pipeline
    // it is still possible that the index for a read in the pipeline does not have its index loaded
    // if the client disconnected and caused the index to be unloaded while there were still reads in
    // the pipeline. For this reason we do not assert a non-null index.
    return get_or_load_compatible_index(index_file, options);
}

bool IndexFileAccess::is_index_loaded(const std::string& index_file,
                                      const Minimap2Options& options) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return is_index_loaded_impl(index_file, options);
}

std::string IndexFileAccess::generate_sequence_records_header(const std::string& index_file,
                                                              const Minimap2Options& options) {
    auto loaded_index = get_index(index_file, options);
    assert(loaded_index && "Index must be loaded to generate header records");
    auto sequence_records = loaded_index->get_sequence_records_for_header();

    std::ostringstream header_stream{};
    bool first_record{true};
    for (const auto& sequence_record : sequence_records) {
        if (!first_record) {
            header_stream << '\n';
        } else {
            first_record = false;
        }
        header_stream << "@SQ\tSN:" << sequence_record.first << "\tLN:" << sequence_record.second;
    }

    return header_stream.str();
}

void IndexFileAccess::unload_index(const std::string& index_file,
                                   const Minimap2IndexOptions& index_options) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_index_lut[{index_file, index_options}] = {};
}

bool validate_options(const Minimap2Options& options) {
    Minimap2Index index{};
    return index.initialise(options);
}

}  // namespace dorado::alignment
