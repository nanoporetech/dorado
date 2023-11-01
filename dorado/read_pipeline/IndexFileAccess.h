#pragma once

#include "ReadPipeline.h"

#include <minimap.h>

#include <memory>
#include <string>

namespace dorado::alignment {

/// <summary>
/// Index file loaded with specific index options
/// </summary>
class Aligner {
    std::shared<mm_idx_t> m_index;
    std::shared<mm_idxopt_t> m_index_options;
    std::string m_reference_file;

public:
    align(BamPtr bam_ptr, const mm_mapopt_t* mapping_options);
};

struct Minimap2IndexOptions {
    short kmer_size;
    short window_size;
    uint64_t index_batch_size;
};

class AbstractIndexLoader {
public:
    virtual bool load_index(const std::string& index_file,
                            const Minimap2IndexOptions* index_options) = 0;
    virtual void unload_index(const std::string& index_file,
                              const Minimap2IndexOptions* index_options) = 0;
};

class Minimap2Aligner {
public:
    align(BamPtr bam_ptr, );
};

class IndexFileAccess : public AbstractIndexLoader {
public:
    bool load_index(const std::string& index_file,
                    const Minimap2IndexOptions* index_options) override;
    void unload_index(const std::string& index_file,
                      const Minimap2IndexOptions* index_options) override;
};

}  // namespace dorado::alignment
