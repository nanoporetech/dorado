#pragma once

#include "Minimap2IndexSupportTypes.h"
#include "Minimap2Options.h"

#include <minimap.h>

#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace dorado::alignment {

class Minimap2Index {
    Minimap2Options m_options;
    std::shared_ptr<const mm_idx_t> m_index;
    std::optional<mm_idxopt_t> m_index_options{};
    std::optional<mm_mapopt_t> m_mapping_options{};

    void set_index_options(const Minimap2IndexOptions& index_options);
    void set_mapping_options(const Minimap2MappingOptions& mapping_options);

    // returns false if a split index
    bool load_index_unless_split(const std::string& index_file, int num_threads);

public:
    bool initialise(Minimap2Options options);
    IndexLoadResult load(const std::string& index_file, int num_threads);

    // Returns a shallow copy of this MinimapIndex with the given mapping options applied.
    // By contract the given indexing options must be identical to those held in this instance
    // and the underlying index must be loaded.
    // If the given mapping options are invalid/incompatible a nullptr will be returned.
    std::shared_ptr<Minimap2Index> create_compatible_index(const Minimap2Options& options) const;

    const mm_idx_t* index() const { return m_index.get(); }
    const mm_idxopt_t& index_options() const;
    const mm_mapopt_t& mapping_options() const;

    HeaderSequenceRecords get_sequence_records_for_header() const;

    // Testability
    const Minimap2Options& get_options() const;
};

}  // namespace dorado::alignment
