#pragma once

#include "Minimap2Index.h"
#include "read_pipeline/ReadPipeline.h"

#include <minimap.h>

#include <memory>
#include <string>
#include <vector>

namespace dorado::alignment {

// Exposed for testability
extern const std::string UNMAPPED_SAM_LINE_STRIPPED;

class Minimap2Aligner {
public:
    /// Construct with a single index.
    Minimap2Aligner(std::shared_ptr<const Minimap2Index> minimap_index)
            : m_minimap_index(std::move(minimap_index)) {}

    /// Align to the full reference (and merge results if index is split).
    std::vector<BamPtr> align(bam1_t* record, mm_tbuf_t* buf);

    /// Align to the full reference (and merge results if index is split).
    void align(dorado::ReadCommon& read_common,
               const std::string& alignment_header,
               mm_tbuf_t* buf);

    /** Get the mapping details from a single alignment.
     *  This function should only be used when either the index is not split, or a split
     *  index is being loaded incrementally. In either case, the Minimap2Aligner object will
     *  contain only a single entry in m_minimap_indexes.
     */
    std::tuple<mm_reg1_t*, int> get_mapping(bam1_t* record, mm_tbuf_t* buf);

    /// This will combine the sequence records from all blocks of a split-index.
    HeaderSequenceRecords get_sequence_records_for_header() const;

private:
    std::shared_ptr<const Minimap2Index> m_minimap_index;

    std::vector<BamPtr> align_impl(bam1_t* record, mm_tbuf_t* buf, int idx_no);
    std::vector<AlignmentResult> align_impl(dorado::ReadCommon& read_common,
                                            const std::string& alignment_header,
                                            mm_tbuf_t* buf,
                                            int idx_no);
    void add_tags(bam1_t*, const mm_reg1_t*, const std::string&, const mm_tbuf_t*, int idx_no);
};

}  // namespace dorado::alignment
