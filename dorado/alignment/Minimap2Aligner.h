#pragma once

#include "Minimap2Index.h"
#include "read_pipeline/ReadPipeline.h"

#include <minimap.h>

#include <memory>
#include <vector>

namespace dorado::alignment {

// Exposed for testability
extern const std::string UNMAPPED_SAM_LINE_STRIPPED;

class Minimap2Aligner {
public:
    Minimap2Aligner(std::shared_ptr<const Minimap2Index> minimap_index)
            : m_minimap_index(std::move(minimap_index)) {}

    void add_tags(bam1_t*, const mm_reg1_t*, const std::string&, const mm_tbuf_t*);
    std::vector<BamPtr> align(bam1_t* record, mm_tbuf_t* buf);
    void align(dorado::ReadCommon& read_common,
               const std::string& alignment_header,
               mm_tbuf_t* buf);
    std::tuple<mm_reg1_t*, int> get_mapping(bam1_t* record, mm_tbuf_t* buf);

    HeaderSequenceRecords get_sequence_records_for_header() const;

private:
    std::shared_ptr<const Minimap2Index> m_minimap_index;
};

}  // namespace dorado::alignment
