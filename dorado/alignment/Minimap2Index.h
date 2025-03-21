#pragma once

#include "Minimap2IndexSupportTypes.h"
#include "Minimap2Options.h"

#include <minimap.h>

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace dorado::alignment {

struct IndexReaderDeleter {
    void operator()(mm_idx_reader_t* index_reader);
};
using IndexReaderPtr = std::unique_ptr<mm_idx_reader_t, IndexReaderDeleter>;

class Minimap2Index {
    Minimap2Options m_options;
    std::vector<std::shared_ptr<const mm_idx_t>> m_indexes;
    IndexReaderPtr m_index_reader;
    bool m_incremental_load{false};

    void set_index(std::shared_ptr<const mm_idx_t> index);
    void add_index(std::shared_ptr<const mm_idx_t> index);

    // Returns nullptr if loading failed, and the load-result, as a pair.
    std::pair<std::shared_ptr<mm_idx_t>, IndexLoadResult> load_initial_index(
            const std::string& index_file,
            int num_threads);

public:
    bool initialise(Minimap2Options options);
    IndexLoadResult load(const std::string& index_file, int num_threads, bool incremental_load);
    IndexLoadResult load_next_chunk(int num_threads);

    /** Returns a shallow copy of this MinimapIndex with the given mapping options applied.
     * By contract the given indexing options must be identical to those held in this instance
     * and the underlying index must be loaded.
     * If the given mapping options are invalid/incompatible a nullptr will be returned.
     */
    std::shared_ptr<Minimap2Index> create_compatible_index(const Minimap2Options& options) const;

    const mm_idx_t* index() const;          ///< Throws if the index is split.
    const mm_idx_t* index(size_t n) const;  ///< Returns the nth block of a split index.
    const mm_idxopt_t& index_options() const;
    const mm_mapopt_t& mapping_options() const;  ///< Should only be called for non-split indexes.

    HeaderSequenceRecords get_sequence_records_for_header() const;
    size_t num_loaded_index_blocks() const { return m_indexes.size(); }

    // Testability
    const Minimap2Options& get_options() const;
};

}  // namespace dorado::alignment
