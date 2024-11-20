#pragma once

#include "htslib/sam.h"

#include <filesystem>

// parameters for bam iteration
typedef struct {
    htsFile* fp;
    sam_hdr_t* hdr;
    hts_itr_t* iter;
    int min_mapQ;
    char tag_name[2];
    int tag_value;
    bool keep_missing;
    const char* read_group;
} mplp_data;

class BamFile {
public:
    BamFile(const std::filesystem::path& in_fn);

    // ~BamFile();

    // Getters.
    htsFile* fp() const { return m_fp.get(); }
    hts_idx_t* idx() const { return m_idx.get(); }
    sam_hdr_t* hdr() const { return m_hdr.get(); }

    htsFile* fp() { return m_fp.get(); }
    hts_idx_t* idx() { return m_idx.get(); }
    sam_hdr_t* hdr() { return m_hdr.get(); }

private:
    std::unique_ptr<htsFile, decltype(&hts_close)> m_fp;
    std::unique_ptr<hts_idx_t, decltype(&hts_idx_destroy)> m_idx;
    std::unique_ptr<sam_hdr_t, decltype(&sam_hdr_destroy)> m_hdr;
};

// // Initialise BAM file, index and header structures
// bam_fset *create_bam_fset(const char *fname);

// // Destory BAM file, index and header structures
// void destroy_bam_fset(bam_fset *fset);

// iterator for reading bam
int read_bam(void* data, bam1_t* b);
