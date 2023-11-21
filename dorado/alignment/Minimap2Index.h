#pragma once

#include "Minimap2IndexSupportTypes.h"
#include "Minimap2Options.h"

#include <minimap.h>

#include <memory>
#include <string>
#include <utility>

namespace dorado::alignment {

class Minimap2Index {
    struct IndexDeleter {
        void operator()(mm_idx_t* index) { mm_idx_destroy(index); }
    };
    using IndexPtr = std::unique_ptr<mm_idx_t, IndexDeleter>;

    IndexPtr m_index;
    mm_idxopt_t m_index_options;
    mm_mapopt_t m_mapping_options;

    void set_index_options(const Minimap2IndexOptions& index_options);
    void set_mapping_options(const Minimap2MappingOptions& mapping_options);

    // returns false if a split index
    bool load_index_unless_split(const std::string& index_file, int num_threads);

public:
    IndexLoadResult load(const std::string& index_file,
                         const Minimap2Options& options,
                         int num_threads);

    const mm_idx_t* index() const { return m_index.get(); }
    const mm_idxopt_t& index_options() const { return m_index_options; }
    const mm_mapopt_t& mapping_options() const { return m_mapping_options; }

    HeaderSequenceRecords get_sequence_records_for_header() const;
};

}  // namespace dorado::alignment
