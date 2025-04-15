#pragma once

#include <cstdint>

struct htsFile;
struct sam_hdr_t;
struct hts_itr_t;
struct bam1_t;

namespace dorado::secondary {

struct HtslibMpileupData {
    htsFile* fp = nullptr;
    sam_hdr_t* hdr = nullptr;
    hts_itr_t* iter = nullptr;
    int32_t min_mapq = 0;
    char tag_name[2] = "";
    int32_t tag_value = 0;
    bool keep_missing = false;
    const char* read_group = nullptr;
};

/**
 * \brief Iterator for reading BAM records during Mpileup calculation.
 *          Only takes primary records.
 *          C-style interface for Htslib.
 */
int32_t mpileup_read_bam(void* data, bam1_t* b);

}  // namespace dorado::secondary
