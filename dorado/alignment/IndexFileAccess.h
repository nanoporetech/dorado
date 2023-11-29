#pragma once

#include "Minimap2IndexSupportTypes.h"
#include "Minimap2Options.h"

#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace dorado::alignment {

class Minimap2Index;

class IndexFileAccess {
    mutable std::mutex m_mutex{};
    using CompatibleIndicesLut = std::map<Minimap2MappingOptions, std::shared_ptr<Minimap2Index>>;
    using IndexKey = std::pair<std::string, Minimap2IndexOptions>;
    std::map<IndexKey, CompatibleIndicesLut> m_index_lut;

    // Retrieves the index if available, will also create the index if a compatible
    // one is already loaded.
    std::shared_ptr<Minimap2Index>& get_index_impl(const std::string& file,
                                                   const Minimap2Options& options);

    // Returns true if the index is loaded, will also create the index if a compatible
    // one is already loaded and return true.
    bool try_load_compatible_index(const std::string& file, const Minimap2Options& options);

    // Requires the mutex to be locked before calling.
    const Minimap2Index* get_compatible_index(const std::string& file,
                                              const Minimap2IndexOptions& indexing_options);

    // Requires the mutex to be locked before calling.
    std::shared_ptr<Minimap2Index> get_exact_index(const std::string& file,
                                                   const Minimap2Options& options);

    // Requires the mutex to be locked before calling.
    bool is_index_loaded_impl(const std::string& file, const Minimap2Options& options) const;

public:
    IndexLoadResult load_index(const std::string& file,
                               const Minimap2Options& options,
                               int num_threads);

    // Testability. Method needed to support utests
    bool is_index_loaded(const std::string& file, const Minimap2Options& options) const;

    // N.B. By contract load_index must be called prior to calling get_index.
    // i.e. will not return nullptr, there will be an assertion failure if load_index
    // has not been called for the file and index options.
    std::shared_ptr<const Minimap2Index> get_index(const std::string& file,
                                                   const Minimap2Options& options);

    // Unloads all compatible indices for the given file and indexing options.
    // The underlying minimap index will be deallocated.
    void unload_index(const std::string& file, const Minimap2IndexOptions& index_options);
};

bool validate_options(const Minimap2Options& options);

}  // namespace dorado::alignment
