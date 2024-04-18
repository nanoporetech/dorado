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

    // Returns true if the index is loaded, will also create the index if a compatible
    // one is already loaded and return true.
    bool try_load_compatible_index(const std::string& index_file, const Minimap2Options& options);

    // By contract the index must be loaded (else assertion failure)
    std::shared_ptr<Minimap2Index> get_exact_index(const std::string& index_file,
                                                   const Minimap2Options& options) const;

    // Requires the mutex to be locked before calling.
    const Minimap2Index* get_compatible_index(const std::string& index_file,
                                              const Minimap2IndexOptions& indexing_options);

    // Requires the mutex to be locked before calling.
    std::shared_ptr<Minimap2Index> get_exact_index_impl(const std::string& index_file,
                                                        const Minimap2Options& options) const;

    // Requires the mutex to be locked before calling.
    bool is_index_loaded_impl(const std::string& index_file, const Minimap2Options& options) const;

    // Requires the mutex to be locked before calling.
    std::shared_ptr<Minimap2Index> get_or_load_compatible_index(const std::string& index_file,
                                                                const Minimap2Options& options);

public:
    IndexLoadResult load_index(const std::string& index_file,
                               const Minimap2Options& options,
                               int num_threads);

    // Returns the index if already loaded, if not loaded will create an index from an
    // existing compatible one.
    // By contract there must be a loaded index for the index file with matching indexing
    // options, if not there will be an assertion failure.
    std::shared_ptr<const Minimap2Index> get_index(const std::string& index_file,
                                                   const Minimap2Options& options);

    // Unloads all compatible indices for the given index file and indexing options.
    // The underlying minimap index will be deallocated.
    void unload_index(const std::string& index_file, const Minimap2IndexOptions& index_options);

    // returns a string containing the sequence records for the requested index
    // Note, if not loaded will create an index from an existing compatible one.
    // By contract there must be a loaded index for the index file with matching indexing
    // options, if not there will be an assertion failure.
    std::string generate_sequence_records_header(const std::string& index_file,
                                                 const Minimap2Options& options);

    // Testability. Method needed to support utests
    bool is_index_loaded(const std::string& index_file, const Minimap2Options& options) const;
};

bool validate_options(const Minimap2Options& options);

}  // namespace dorado::alignment
